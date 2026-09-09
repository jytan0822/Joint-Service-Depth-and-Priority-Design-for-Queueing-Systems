"""Offline regression tests for the notebook's strict-only SGD scorer.

Only function definitions and SCORING_VERSION are extracted from the scoring
cell. The notebook runner and API client are never imported or executed.
Run with: python -m unittest test_sgd_strict_scoring -v
"""

import ast
from collections import Counter
import copy
import json
from pathlib import Path
import re
import unittest
import unicodedata


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "SGD_service_depth_experiment.ipynb"
RESULTS = ROOT / "results"
EXPECTED_COUNTS = {"none": 77, "low": 86, "medium": 90, "high": 92, "xhigh": 93}
REMOVED_FIELDS = {
    "value",
    "resolved_exact",
    "slot_values_exact",
    "individual_slot_results",
    "semantic_correct",
    "semantic_match_threshold",
    "best_similarity",
    "valuation_version",
}


def load_scorer():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    scoring_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "def score_sgd_prediction(" in "".join(cell["source"])
    ]
    if len(scoring_cells) != 1:
        raise AssertionError("Expected exactly one scoring cell.")
    tree = ast.parse(scoring_cells[0])
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SCORING_VERSION"
                for target in node.targets
            )
        )
    ]
    schemas = json.loads(
        (ROOT / "data" / "sgd" / "test" / "schema.json").read_text(encoding="utf-8")
    )
    namespace = {
        "re": re,
        "unicodedata": unicodedata,
        "schemas_by_split": {
            "test": {schema["service_name"]: schema for schema in schemas}
        },
    }
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(NOTEBOOK) + ":strict_scorer", "exec"), namespace)
    return namespace, scoring_cells[0]


def read_runs(size):
    matches = list(RESULTS.glob(f"sgd_test_*_size{size}_*_v1_runs.jsonl"))
    if len(matches) != 1:
        raise AssertionError(f"Expected one canonical size-{size} result log.")
    return [json.loads(line) for line in matches[0].read_text(encoding="utf-8").splitlines() if line.strip()]


class StrictScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.source = load_scorer()

    def setUp(self):
        self.gold = {
            "service": "Alarm_1",
            "active_intent": "GetAlarms",
            "slot_values": {"alarm_name": ["Morning Alarm"], "alarm_time": ["7 am"]},
        }
        self.prediction = {
            "service": "Alarm_1",
            "active_intent": "GetAlarms",
            "slot_values": [
                {"slot": "alarm_name", "values": ["Morning Alarm"]},
                {"slot": "alarm_time", "values": ["7 am"]},
            ],
        }

    def score(self, prediction=None, gold=None):
        return self.namespace["score_sgd_prediction"](
            self.prediction if prediction is None else prediction,
            self.gold if gold is None else gold,
            "test",
        )

    def assert_strict_only(self, item):
        if isinstance(item, dict):
            self.assertFalse(REMOVED_FIELDS.intersection(item), REMOVED_FIELDS.intersection(item))
            for value in item.values():
                self.assert_strict_only(value)
        elif isinstance(item, list):
            for value in item:
                self.assert_strict_only(value)

    def test_exact_match_and_scoring_version(self):
        scores = self.score()
        self.assertEqual(scores["resolved_strict_exact"], 1)
        self.assertEqual(scores["slot_values_strict_exact"], 1)
        self.assertTrue(all(scores["individual_slot_results_strict"].values()))
        self.assertEqual(self.namespace["SCORING_VERSION"], "sgd_strict_exact_v1")
        self.assert_strict_only(scores)

    def test_minimal_text_normalization(self):
        self.prediction["slot_values"][0]["values"] = ["  ＭＯＲＮＩＮＧ\t\nALARM  "]
        self.prediction["slot_values"][1]["values"] = [" ７  AM "]
        self.assertEqual(self.score()["resolved_strict_exact"], 1)

    def test_any_annotated_gold_alternative_is_accepted(self):
        self.gold["slot_values"]["alarm_name"] = ["Morning Alarm", "Wake-up"]
        self.prediction["slot_values"][0]["values"] = ["Wake-up"]
        self.assertEqual(self.score()["resolved_strict_exact"], 1)

    def test_no_fuzzy_token_order_or_numeric_semantic_shortcuts(self):
        for slot, predicted, gold in [
            ("alarm_name", "Morning Alarn", "Morning Alarm"),
            ("alarm_name", "Alarm Morning", "Morning Alarm"),
            ("alarm_name", "The Morning Alarm", "Morning Alarm"),
            ("alarm_time", "seven am", "7 am"),
            ("alarm_time", "7:00 am", "7 am"),
            ("alarm_time", "8 am", "7 am"),
        ]:
            with self.subTest(predicted=predicted, gold=gold):
                prediction = copy.deepcopy(self.prediction)
                gold_state = copy.deepcopy(self.gold)
                gold_state["slot_values"][slot] = [gold]
                next(item for item in prediction["slot_values"] if item["slot"] == slot)["values"] = [predicted]
                self.assertEqual(self.score(prediction, gold_state)["resolved_strict_exact"], 0)

    def test_missing_slot_fails(self):
        self.prediction["slot_values"].pop()
        self.assertEqual(self.score()["resolved_strict_exact"], 0)

    def test_extra_slot_fails_even_if_schema_valid(self):
        self.prediction["slot_values"].append({"slot": "new_alarm_name", "values": ["Other"]})
        scores = self.score()
        self.assertEqual(scores["slot_names_valid"], 1)
        self.assertEqual(scores["resolved_strict_exact"], 0)

    def test_duplicate_slot_fails(self):
        self.prediction["slot_values"].append(copy.deepcopy(self.prediction["slot_values"][0]))
        self.assertEqual(self.score()["resolved_strict_exact"], 0)

    def test_exactly_one_predicted_value_is_required(self):
        for values in [[], ["Morning Alarm", "Wake-up"], ["Morning Alarm", "Morning Alarm"]]:
            with self.subTest(values=values):
                self.prediction["slot_values"][0]["values"] = values
                self.assertEqual(self.score()["resolved_strict_exact"], 0)

    def test_wrong_service_and_intent_fail(self):
        self.prediction["service"] = "Events_3"
        scores = self.score()
        self.assertEqual(scores["service_correct"], 0)
        self.assertEqual(scores["resolved_strict_exact"], 0)
        self.prediction["service"] = "Alarm_1"
        self.prediction["active_intent"] = "AddAlarm"
        scores = self.score()
        self.assertEqual(scores["intent_correct"], 0)
        self.assertEqual(scores["service_intent_valid"], 1)
        self.assertEqual(scores["resolved_strict_exact"], 0)

    def test_unknown_service_intent_and_slot_fail_schema_checks(self):
        self.prediction["service"] = "UnknownService"
        scores = self.score()
        self.assertEqual(scores["service_intent_valid"], 0)
        self.assertEqual(scores["resolved_strict_exact"], 0)
        self.prediction["service"] = "Alarm_1"
        self.prediction["active_intent"] = "UnknownIntent"
        self.assertEqual(self.score()["service_intent_valid"], 0)
        self.prediction["active_intent"] = "GetAlarms"
        self.prediction["slot_values"].append({"slot": "unknown_slot", "values": ["unknown"]})
        scores = self.score()
        self.assertEqual(scores["slot_names_valid"], 0)
        self.assertEqual(scores["resolved_strict_exact"], 0)

    def test_replay_all_size200_completed_responses(self):
        rows = [row for row in read_runs(200) if row.get("success")]
        self.assertEqual(len(rows), 1000)
        totals = Counter()
        counts = Counter()
        keys = set()
        for row in rows:
            key = (row["case_id"], row["reasoning_effort"], row["repetition"])
            self.assertNotIn(key, keys)
            keys.add(key)
            scores = self.score(row["prediction"], row["gold_state"])
            with self.subTest(case=key):
                for field in ("service_correct", "intent_correct", "slot_values_strict_exact", "resolved_strict_exact"):
                    self.assertEqual(scores[field], row[field])
                    self.assertEqual(scores[field], row["scores"][field])
                self.assertEqual(scores["individual_slot_results_strict"], row["scores"]["individual_slot_results_strict"])
            effort = row["reasoning_effort"]
            totals[effort] += scores["resolved_strict_exact"]
            counts[effort] += 1
        self.assertEqual(dict(totals), EXPECTED_COUNTS)
        self.assertEqual(dict(counts), {effort: 200 for effort in EXPECTED_COUNTS})

    def test_canonical_result_log_is_strict_only(self):
        for row in read_runs(200):
            with self.subTest(attempt=row.get("attempt_id")):
                self.assert_strict_only(row)
                self.assertEqual(row["scoring_version"], self.namespace["SCORING_VERSION"])
                if row.get("success"):
                    self.assertIn("resolved_strict_exact", row)
                    self.assertIn("resolved_strict_exact", row["scores"])
                    self.assertEqual(
                        self.score(row["prediction"], row["gold_state"])["resolved_strict_exact"],
                        row["resolved_strict_exact"],
                    )

    def test_semantic_rescored_copies_and_scorer_are_removed(self):
        self.assertEqual(list(RESULTS.glob("*_semantic_rescored.jsonl")), [])
        for removed in ("SequenceMatcher", "SEMANTIC_MATCH_THRESHOLD", "def normalize_semantic_value", "def noncategorical_similarity", "def numeric_signature", "VALUATION_VERSION"):
            self.assertNotIn(removed, self.source)


if __name__ == "__main__":
    unittest.main()
