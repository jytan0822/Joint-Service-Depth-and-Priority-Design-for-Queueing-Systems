# Joint Service Depth and Priority Design in Queueing Systems

This repository contains the official source code to reproduce the numerical examples and the empirical case study presented in the research paper **"Joint Service Depth and Priority Design in Queueing Systems"**.

---

## 🛠️ Experimental Setup & Solvers

The notebooks run in a local Python environment; no GPU or cluster is required. The theory notebooks use `SciPy` (`differential_evolution`, `SLSQP`, `brentq`) to solve the welfare-maximization problems, and the case-study notebooks evaluate exact stationary M/G/1 formulas in `NumPy` with no simulation and no API calls.

| Package | Version |
| :--- | :--- |
| Python | 3.13.13 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.10.9 |
| OpenAI | 3.3.1 (data-collection notebook only) |

```bash
conda create -n research -c conda-forge python=3.13 numpy=2.4 scipy=1.17 matplotlib=3.10 jupyter
conda activate research
pip install openai==3.3.1   # only needed to re-run the paid data collection
```

Earlier NumPy/SciPy releases work as well; the versions above are the ones the reported results were produced with.

Every notebook writes its figures to paths **relative to the working directory**, so launch Jupyter from the folder containing the notebook you intend to run.

---

## 📊 Notebook Directory

### 1. Theoretical Insights & Propositions

These notebooks reproduce the theoretical visualizations and the welfare-gap tables discussed in the paper.

| Figure / Table | Notebook | Description |
| :--- | :--- | :--- |
| **Figure 1** | `Proposition_1.ipynb` | Priority-switch threshold $c$ and welfare under each priority rule (Proposition 1). |
| **Tables 1 & 2** | `Section_6.1.ipynb` | Welfare gap across $\lambda_2/\lambda_1$ and across $(\theta_1,\theta_2)$, for four value functions. |
| **Figures 2 & 3** | `Section_6.2.ipynb` | Optimal service depth, sojourn time, welfare/surplus and payment for a continuum of types. |

`Proposition_1.ipynb` spends almost all of its runtime in the threshold-curve cell, which solves both priority regimes at 35 values of $\theta_1^2/(a\theta_2)$; the plotting cell reads the cached `curve` object, so figures can be restyled without recomputing.

`Section_6.2.ipynb` produces one value function per run. It ships configured for $V(x)=\ln(1+x)$ (Figure 3); to produce Figure 2, set `U` to `np.sqrt` in the first cell and change the four `savefig` filenames from the `log_` prefix to `sqrt_`.

### 2. Empirical Case Study

Evaluations using an AI agent's measured service values and service times, with reasoning effort as the service depth. All notebooks in this section read the saved run log and are fully offline.

| Paper Section | Notebook | Description |
| :--- | :--- | :--- |
| **Section 7.1** | `Case Study/SGD_service_depth_experiment.ipynb` | Samples 200 SGD test turns and queries the agent at five reasoning efforts. **Paid API calls; see the caveat below.** |
| **Figure 4, Section 7.2** | `Case Study/SGD_stats.ipynb` | Empirical calibration: service values $V(d_k)$, mean service times $S(d_k)$, and the service-time histograms. |
| **Figure 5, Section 7.3** | `Case Study/two_type_load.ipynb` | Two-type model: optimal policy, optimal IC menu and welfare as the arrival rate $\lambda$ varies. |
| **Figure 6, Section 7.3** | `Case Study/two_type_composition.ipynb` | Two-type model: the same objects as the patient-to-impatient ratio $\lambda_2/\lambda_1$ varies. |
| **Figure 7, Section 7.4** | `Case Study/continuous_type_load.ipynb` | Value of joint design: welfare, service depth and sojourn time under four designs. |
| **Figure 8, Section 7.4** | `Case Study/continuous_type_personalization.ipynb` | Value of personalization: a two-cluster population treated as two types or as a continuum. |

Supporting modules in `Case Study/`:

| File | Purpose |
| :--- | :--- |
| `sgd_calibration.py` | Loads the run log into aligned request-by-depth arrays and returns the normalized queueing moments $V(d_k)$, $S(d_k)$ and $\mathbb{E}[S^2(d_k)]$. Every downstream notebook calls `load_calibration()`. |
| `test_sgd_calibration.py` | Regression tests for the calibration loader and its cost accounting. |
| `test_sgd_strict_scoring.py` | Replays all 1,000 logged responses through the strict scorer and checks the recorded labels. |

```bash
cd "Case Study"
python -m unittest test_sgd_calibration test_sgd_strict_scoring   # 19 tests
```

---

## ⚠️ Re-running the Data Collection

`SGD_service_depth_experiment.ipynb` is included for transparency and **is not part of the reproduction path**. Running it requires an `OPENAI_API_KEY` and incurs paid API calls, and model responses are not deterministic, so it will not reproduce the saved log exactly. It is guarded by `RUN_PAID_EXPERIMENT = False`.

The source of truth for every result in the paper is the saved run log:

```text
Case Study/results/sgd_test_request_state_actionable_size200_seed20260822_unique_datapoints_sgd_request_state_v1_runs.jsonl
```

It holds 1,015 attempts, of which 1,000 are completed responses covering 200 distinct user turns at five reasoning efforts (`none`, `low`, `medium`, `high`, `xhigh`). The calibration uses the final completed response for each (turn, depth) pair; completed-but-incorrect responses still contribute a service time. Two details of the collection are worth recording: the recorded model is `gpt-5.6-luna`, and 769 of the calls were issued with `max_output_tokens=4000` before the cap was raised to 10,000 for the remainder (only one completed response exceeded 4,000 output tokens, so the calibration is unaffected).

Service time is a cost-based resource proxy, $S_{jk}=\kappa C_{jk}$, with $\kappa$ chosen so the pooled mean over all 1,000 responses equals one. It is not measured wall-clock inference time.

---

## 💾 Case Study Data Setup

The case study uses the Schema-Guided Dialogue (SGD) dataset of *Rastogi et al. (2020)*, released under CC BY-SA 4.0. Only the **test split** is read (`SPLITS = ("test",)`).

Clone the dataset and place its `test` directory, together with the dataset licence, under `Case Study/data/sgd/`:

```text
Code/
├── Case Study/
│   ├── SGD_service_depth_experiment.ipynb
│   └── data/
│       └── sgd/
│           ├── LICENSE.txt
│           └── test/
│               ├── schema.json
│               └── dialogues_001.json ... dialogues_034.json
```

```bash
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git
mkdir -p "Case Study/data/sgd"
cp -r dstc8-schema-guided-dialogue/test "Case Study/data/sgd/"
cp dstc8-schema-guided-dialogue/LICENSE.txt "Case Study/data/sgd/"
```

The `train` and `dev` splits are not used and do not need to be downloaded.

---

## 📁 Repository Layout

```text
Code/
├── README.md
├── Proposition_1.ipynb                     # Figure 1
├── Section_6.1.ipynb                       # Tables 1 and 2
├── Section_6.2.ipynb                       # Figures 2 and 3
├── prop1_1.pdf, prop1_2.pdf                # Figure 1 panels
├── sqrt_*.pdf                              # Figure 2 panels
├── log_*.pdf                               # Figure 3 panels
└── Case Study/
    ├── SGD_service_depth_experiment.ipynb  # data collection (paid)
    ├── SGD_stats.ipynb                     # Figure 4
    ├── two_type_load.ipynb                 # Figure 5
    ├── two_type_composition.ipynb          # Figure 6
    ├── continuous_type_load.ipynb          # Figure 7
    ├── continuous_type_personalization.ipynb  # Figure 8
    ├── sgd_calibration.py
    ├── test_sgd_calibration.py
    ├── test_sgd_strict_scoring.py
    ├── data/sgd/test/                      # SGD test split (see above)
    └── results/                            # run log, calibration inputs, generated figures
```

Theory figures are written to `Code/`; case-study figures are written to `Code/Case Study/results/`. The paper's LaTeX source expects them under `Figures/` and `Figures/Case study/` respectively, so figures are copied into the manuscript tree after regeneration.
