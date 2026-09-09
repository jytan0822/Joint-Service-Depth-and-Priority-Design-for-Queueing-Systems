"""Read-only calibration and paired dialogue-cluster uncertainty for SGD.

No API calls are made and the source experiment log is never modified.  The
estimand is the outcome and resource usage of the final completed response for
each sampled request/effort pair, conditional on the inherited experiment.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np


EFFORTS = ("none", "low", "medium", "high", "xhigh")
DEFAULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "sgd_test_request_state_actionable_size200_seed20260822_unique_datapoints_"
    "sgd_request_state_v1_runs.jsonl"
)
PROXIES = {
    "cost": "estimated API expenditure",
    "output_tokens": "output tokens, including reasoning tokens",
    "total_tokens": "input plus output tokens",
}


def _finite_number(record, field, *, allow_zero=True):
    value = record.get(field)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Missing or invalid {field} for {record.get('case_id')}")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"Invalid {field}={value} for {record.get('case_id')}")
    return value


def _dialogue_id(case_id):
    match = re.fullmatch(r"(.+)__(.+)__turn_(\d+)", case_id)
    if match is None:
        raise ValueError(f"Cannot recover the dialogue ID from {case_id!r}")
    return match.group(1) + "__" + match.group(2)


def _moments(work, outcomes, coefficient=None):
    if coefficient is None:
        coefficient = 1.0 / float(np.mean(work))
    coefficient = float(coefficient)
    if not np.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("The service conversion coefficient must be positive.")
    times = work * coefficient
    return {
        "coefficient": coefficient,
        "times": times,
        "values": np.mean(outcomes, axis=0),
        "means": np.mean(times, axis=0),
        # 'seconds' denotes the second raw moment, not physical seconds.
        "seconds": np.mean(times * times, axis=0),
    }


def load_calibration(path=None, proxy="cost"):
    """Return aligned request-by-effort arrays and normalized queue moments.

    Accepted work proxies are ``cost``, ``output_tokens`` and ``total_tokens``.
    All have overall empirical mean service requirement one. ``seconds`` is
    the vector of second raw service moments E[S^2], never wall-clock time.
    Duplicate completed pairs, if present, use their last log entry; incomplete
    effort panels are rejected rather than silently discarded.
    """
    if proxy not in PROXIES:
        raise ValueError(f"proxy must be one of {tuple(PROXIES)}")
    path = Path(path or DEFAULT_PATH).resolve()
    content = path.read_bytes()
    records = []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    selected = {}
    completed_count = 0
    for record in records:
        if record.get("success") is not True:
            continue
        if record.get("response_status") != "completed":
            raise ValueError("A success record is not marked completed.")
        completed_count += 1
        effort = record.get("reasoning_effort")
        if effort not in EFFORTS:
            raise ValueError(f"Unexpected completed reasoning effort: {effort!r}")
        key = (str(record["case_id"]), effort)
        selected[key] = record
    case_ids = sorted({key[0] for key in selected})
    if not case_ids:
        raise ValueError("The experiment contains no completed responses.")
    missing = [(case, effort) for case in case_ids for effort in EFFORTS
               if (case, effort) not in selected]
    if missing:
        raise ValueError(f"Incomplete paired effort panels: {missing[:5]}")
    flat = [selected[(case, effort)] for case in case_ids for effort in EFFORTS]
    n = len(case_ids)

    def matrix(field, allow_zero=True):
        return np.asarray([
            _finite_number(record, field, allow_zero=allow_zero) for record in flat
        ], dtype=float).reshape(n, len(EFFORTS))

    outcomes = matrix("resolved_strict_exact")
    if not np.all(np.isin(outcomes, [0.0, 1.0])):
        raise ValueError("resolved_strict_exact must be a binary indicator.")
    costs = matrix("estimated_cost_usd", allow_zero=False)
    input_tokens = matrix("input_tokens")
    output_tokens = matrix("output_tokens", allow_zero=False)
    reasoning_tokens = matrix("reasoning_tokens")
    visible_output_tokens = matrix("visible_output_tokens")
    total_tokens = matrix("total_tokens", allow_zero=False)
    if not np.array_equal(total_tokens, input_tokens + output_tokens):
        raise ValueError("The recorded total token counts do not equal input plus output.")
    if not np.array_equal(output_tokens, reasoning_tokens + visible_output_tokens):
        raise ValueError("Reasoning plus visible tokens do not equal total output tokens.")

    def input_expenditure(record):
        pricing = record["pricing"]
        return (
            _finite_number(record, "uncached_input_tokens") * pricing["input_per_million_usd"]
            + _finite_number(record, "cached_input_tokens") * pricing["cached_input_per_million_usd"]
            + _finite_number(record, "cache_write_tokens") * pricing["cache_write_per_million_usd"]
        ) / 1e6

    input_costs = np.asarray([input_expenditure(r) for r in flat]).reshape(n, -1)
    output_costs = np.asarray([
        _finite_number(r, "output_tokens") * r["pricing"]["output_per_million_usd"] / 1e6
        for r in flat
    ]).reshape(n, -1)
    if not np.allclose(costs, input_costs + output_costs, rtol=1e-12, atol=1e-14):
        raise ValueError("The logged cost does not match its recorded token-price calculation.")
    work = {"cost": costs, "output_tokens": output_tokens, "total_tokens": total_tokens}[proxy]
    dialogue_ids = np.asarray([_dialogue_id(case) for case in case_ids])
    counts = Counter(dialogue_ids)
    prompt_hash_consistent = all(
        len({selected[(case, effort)]["prompt_sha256"] for effort in EFFORTS}) == 1
        for case in case_ids
    )
    metadata = {
        "source_path": str(path),
        "source_sha256": sha256(content).hexdigest(),
        "n_log_records": len(records),
        "n_completed_attempts": completed_count,
        "n_failed_attempts": len(records) - completed_count,
        "n_selected_completed_pairs": len(flat),
        "n_superseded_completed_pairs": completed_count - len(flat),
        "n_cases": n,
        "n_dialogues": len(counts),
        "n_dialogues_with_multiple_sampled_turns": sum(v > 1 for v in counts.values()),
        "largest_sampled_dialogue_cluster": max(counts.values()),
        "cluster_size_counts": dict(sorted(Counter(counts.values()).items())),
        "proxy": proxy,
        "proxy_description": PROXIES[proxy],
        "normalization": "The pooled mean over requests and all five efforts equals one.",
        "strict_outcome_field": "resolved_strict_exact",
        "completed_pair_selection": "Last completed record per request and effort in log order.",
        "prompt_hash_consistent_within_case": prompt_hash_consistent,
        "requested_models": sorted({r["requested_model"] for r in flat}),
        "returned_models": sorted({r["returned_model"] for r in flat}),
        "service_tiers": sorted({r["service_tier"] for r in flat}),
        "prompt_versions": sorted({r["prompt_version"] for r in flat}),
        "dataset_splits": sorted({r["dataset_split"] for r in flat}),
        "output_cap_completed_counts": dict(sorted(Counter(r["max_output_tokens"] for r in flat).items())),
        "completed_responses_above_4000_output_tokens": int(np.count_nonzero(output_tokens > 4000)),
        "pricing_snapshots": sorted({r["pricing"]["snapshot_date"] for r in flat}),
        "mean_cost_usd": float(np.mean(costs)),
        "mean_input_tokens": float(np.mean(input_tokens)),
        "input_share_of_total_estimated_cost": float(input_costs.sum() / costs.sum()),
        "inference_scope": (
            "Conditional on the sampled actionable SGD test turns, this model/prompt, "
            "and final completed calls; one observed response per request-effort pair. "
            "Resampling does not estimate within-prompt model randomness or retry-inclusive resource usage."
        ),
    }
    result = {
        "case_ids": np.asarray(case_ids),
        "dialogue_ids": dialogue_ids,
        "efforts": EFFORTS,
        "outcomes": outcomes,
        "costs": costs,
        "cost_matrix": costs,
        "input_costs": input_costs,
        "output_costs": output_costs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_output_tokens": visible_output_tokens,
        "total_tokens": total_tokens,
        "services": np.asarray([selected[(case, EFFORTS[0])]["gold_service"] for case in case_ids]),
        "work": work,
        "proxy": proxy,
        "metadata": metadata,
        "source_sha256": metadata["source_sha256"],
    }
    result.update(_moments(work, outcomes))
    return result


def calibration_from_indices(calibration, indices, coefficient=None):
    """Build a paired subset, optionally retaining a training-set time scale.

    With coefficient=None the subset is normalized to pooled mean one.  Pass
    a training coefficient when scoring held-out observations so training and
    evaluation use the same units; renormalizing held-out data would change
    the offered load rather than merely evaluate the trained policy.
    """
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or not indices.size:
        raise ValueError("indices must be a nonempty one-dimensional array.")
    n = len(calibration["case_ids"])
    if np.any(indices < 0) or np.any(indices >= n):
        raise IndexError("Calibration subset index is out of range.")
    result = dict(calibration)
    row_fields = (
        "case_ids", "dialogue_ids", "outcomes", "costs", "cost_matrix",
        "input_costs", "output_costs", "input_tokens", "output_tokens",
        "reasoning_tokens", "visible_output_tokens", "total_tokens",
        "services", "work", "times",
    )
    for key in row_fields:
        result[key] = calibration[key][indices].copy()
    result.update(_moments(result["work"], result["outcomes"], coefficient))
    result["metadata"] = dict(calibration["metadata"])
    counts = Counter(result["dialogue_ids"])
    result["metadata"].update({
        "n_cases": int(len(indices)),
        "n_dialogues": len(counts),
        "n_dialogues_with_multiple_sampled_turns": sum(v > 1 for v in counts.values()),
        "largest_sampled_dialogue_cluster": max(counts.values()),
        "cluster_size_counts": dict(sorted(Counter(counts.values()).items())),
        "subset_of_original_calibration": True,
        "fixed_conversion_coefficient": coefficient is not None,
    })
    return result


def bootstrap_calibrations(calibration, n_boot=1000, seed=20260822, chunk_size=128):
    """Resample complete sampled dialogues with replacement, preserving pairing.

    Each replicate draws G of the G unique sampled dialogues.  Every sampled
    turn within a selected dialogue is included, so replicate request counts
    can differ.  Means weight turns equally, matching the original request-
    level estimand.  Each replicate independently recomputes its pooled work
    normalization, propagating uncertainty in the cost-to-time coefficient.
    """
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer.")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    clusters, inverse = np.unique(calibration["dialogue_ids"], return_inverse=True)
    g = len(clusters)
    if g < 2:
        raise ValueError("At least two dialogues are required for cluster resampling.")
    sizes = np.bincount(inverse).astype(float)
    sums = {}
    for name, matrix in (("outcomes", calibration["outcomes"]),
                         ("work", calibration["work"]),
                         ("work_squared", calibration["work"] ** 2)):
        grouped = np.zeros((g, len(EFFORTS)), dtype=float)
        np.add.at(grouped, inverse, matrix)
        sums[name] = grouped
    rng = np.random.default_rng(seed)
    cluster_counts = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot)
    result = {
        "values": np.empty((n_boot, len(EFFORTS))),
        "means": np.empty((n_boot, len(EFFORTS))),
        "seconds": np.empty((n_boot, len(EFFORTS))),
        "coefficients": np.empty(n_boot),
        "draw_size": cluster_counts @ sizes,
        "cluster_counts": cluster_counts,
        "cluster_ids": clusters,
        "metadata": {
            "n_boot": int(n_boot), "seed": int(seed), "n_clusters": g,
            "resampling_unit": "dialogue; all sampled turns and effort pairs retained together",
            "normalization": "Pooled mean service requirement renormalized to one in every replicate.",
            "interval_type": "pointwise percentile dialogue-cluster bootstrap",
            "inference_scope": calibration["metadata"]["inference_scope"],
        },
    }
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        counts = cluster_counts[start:stop]
        denominator = result["draw_size"][start:stop, None]
        work_means = counts @ sums["work"] / denominator
        coefficients = 1.0 / work_means.mean(axis=1)
        result["values"][start:stop] = counts @ sums["outcomes"] / denominator
        result["means"][start:stop] = work_means * coefficients[:, None]
        result["seconds"][start:stop] = (
            (counts @ sums["work_squared"] / denominator) * coefficients[:, None] ** 2
        )
        result["coefficients"][start:stop] = coefficients
    return result


def calibration_tables(calibration, bootstrap=None, confidence=0.95):
    """Return portable list-of-records tables; no plotting or file writes."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one.")
    tails = [(1 - confidence) / 2, (1 + confidence) / 2]

    def interval(values):
        lower, upper = np.quantile(values, tails)
        return float(lower), float(upper)

    summary = []
    components = []
    for k, effort in enumerate(EFFORTS):
        row = {
            "depth": f"d{k}", "effort": effort,
            "n_requests": len(calibration["case_ids"]),
            "strict_correct": int(calibration["outcomes"][:, k].sum()),
            "strict_accuracy": float(calibration["values"][k]),
            "mean_cost_usd": float(calibration["costs"][:, k].mean()),
            "mean_service": float(calibration["means"][k]),
            "second_service_moment": float(calibration["seconds"][k]),
            "service_sd": float(calibration["times"][:, k].std(ddof=1)),
            "accuracy_per_mean_service": float(calibration["values"][k] / calibration["means"][k]),
        }
        if bootstrap is not None:
            for column, samples in (
                ("strict_accuracy", bootstrap["values"][:, k]),
                ("mean_service", bootstrap["means"][:, k]),
                ("second_service_moment", bootstrap["seconds"][:, k]),
                ("accuracy_per_mean_service", bootstrap["values"][:, k] / bootstrap["means"][:, k]),
            ):
                row[column + "_ci_low"], row[column + "_ci_high"] = interval(samples)
        summary.append(row)
        components.append({
            "effort": effort,
            "mean_input_tokens": float(calibration["input_tokens"][:, k].mean()),
            "mean_output_tokens_including_reasoning": float(calibration["output_tokens"][:, k].mean()),
            "mean_reasoning_tokens": float(calibration["reasoning_tokens"][:, k].mean()),
            "mean_visible_output_tokens": float(calibration["visible_output_tokens"][:, k].mean()),
            "mean_input_cost_usd": float(calibration["input_costs"][:, k].mean()),
            "mean_output_cost_usd": float(calibration["output_costs"][:, k].mean()),
            "input_share_of_total_cost": float(calibration["input_costs"][:, k].sum() / calibration["costs"][:, k].sum()),
        })
    gains = []
    for lower, higher in combinations(range(len(EFFORTS)), 2):
        delta = calibration["outcomes"][:, higher] - calibration["outcomes"][:, lower]
        row = {
            "lower_effort": EFFORTS[lower], "higher_effort": EFFORTS[higher],
            "paired_accuracy_gain": float(delta.mean()),
            "higher_only_correct": int(np.count_nonzero(delta == 1)),
            "lower_only_correct": int(np.count_nonzero(delta == -1)),
            "both_correct": int(np.count_nonzero(
                (calibration["outcomes"][:, higher] == 1) & (calibration["outcomes"][:, lower] == 1))),
        }
        if bootstrap is not None:
            samples = bootstrap["values"][:, higher] - bootstrap["values"][:, lower]
            row["ci_low"], row["ci_high"] = interval(samples)
            # Descriptive bootstrap frequency, not a p-value or posterior probability.
            row["bootstrap_fraction_positive"] = float(np.mean(samples > 0))
        gains.append(row)
    return {
        "calibration": summary,
        "paired_gains": gains,
        "cost_components": components,
        "data_audit": [dict(calibration["metadata"])],
    }
