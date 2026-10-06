# Joint Service Depth and Priority Design in Queueing Systems

Code that reproduces the numerical results in the paper *Joint Service Depth and Priority Design in Queueing Systems* by Adam N. Elmachtoub, Yash Kanoria, and Jonathan Y. Tan.

Each table and figure in the paper comes from one Jupyter notebook. The notebooks evaluate exact queueing formulas or solve small optimization problems. They need no GPU, no simulation and no paid services. The case study uses public benchmark data, which you download with one script.

## Quick start

1. **Install Python and the packages.** The results were produced with Python 3.13.13 and these package versions:

   ```bash
   python -m venv .venv
   source .venv/bin/activate              # Windows: .venv\Scripts\activate
   pip install numpy==2.4.6 scipy==1.17.1 matplotlib==3.10.9 jupyter
   ```

2. **Download the case-study data.** This step is needed only for the notebooks in `Case Study DeepSWE/`, and it requires an internet connection:

   ```bash
   cd "Case Study DeepSWE"
   python prepare_data.py
   cd ..
   ```

3. **Run a notebook.** Open it in Jupyter (`jupyter lab`, then *Run > Run All Cells*), or run it from the command line:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace Section_6.1.ipynb
   ```

   The notebooks are independent and can be run in any order. Each one prints its results and saves its figures as listed below.

## What each notebook reproduces

| Paper | Notebook | Output |
| :--- | :--- | :--- |
| Figure 1 (Section 4.1) | `Proposition_1.ipynb` | `prop1_1.pdf`, `prop1_2.pdf` |
| Tables 1 and 2 (Section 6.1) | `Section_6.1.ipynb` | Printed table rows |
| Figures 2 and 3 (Section 6.2) | `Section_6.2.ipynb` | `sqrt_*.pdf` (Figure 2), `log_*.pdf` (Figure 3) |
| Table 3, Figures 4 and 5 (Section 7.2) | `Case Study DeepSWE/DeepSWE_stats.ipynb` | Printed table; `results/service_value_by_effort.pdf`, `results/service_time_by_effort.pdf`, `results/service_time_histograms.png` |
| Figure 6 (Section 7.3) | `Case Study DeepSWE/two_type_load.ipynb` | `results/two_type_load_*.pdf` |
| Figure 7 (Section 7.3) | `Case Study DeepSWE/two_type_composition.ipynb` | `results/two_type_composition_*.pdf` |
| Figure 8 (Section 7.4) | `Case Study DeepSWE/continuous_type_load.ipynb` | `results/continuous_type_load_*.pdf` |
| Figure 9 (Section 7.4) | `Case Study DeepSWE/continuous_type_personalization.ipynb` | `results/continuous_type_personalization_*.pdf` |
| Table 4 (Section 8.3) | `static_vs_threshold.ipynb` | Printed summary |

The theory notebooks save their figures next to the notebook. The case-study notebooks save theirs in `Case Study DeepSWE/results/`. File names match the ones used in the paper.

## Notes on individual notebooks

**`Proposition_1.ipynb`.** One cell computes the priority-switch threshold curve at 35 values of $\theta_1^2/(a\theta_2)$. The plotting cells reuse that result, so you can restyle the figures without computing the curve again.

**`Section_6.1.ipynb`.** Prints each row of Tables 1 and 2 in the LaTeX format used in the paper. The optimizer uses fixed random seeds, so the results are the same on every run.

**`Section_6.2.ipynb`.** Produces one figure per run. As shipped, it uses $V(x)=\log(1+x)$ and writes the four panels of Figure 3 (`log_*.pdf`). To produce Figure 2, which uses $V(x)=\sqrt{x}$:

1. In the first code cell, comment out `return np.log1p(np.maximum(x, 0.0))` and uncomment `return np.sqrt(np.maximum(x, 0.0))`.
2. In the four plotting cells, change the prefix in each `fig.savefig("log_...")` call from `log_` to `sqrt_`.
3. Run all cells again.

**`static_vs_threshold.ipynb`.** Compares the best static policy with the best threshold policy, which turns away arrivals when the expected workload in the system reaches a threshold $N$. Threshold policies are evaluated exactly, from the stationary distribution of the finite Markov chain on the numbers of customers of each type. For each ratio $\lambda_2/\lambda_1$, the notebook prints the optimal policies. The summary at the end gives the four rows of Table 4:

| Summary column | Table 4 row |
| :--- | :--- |
| `static gap` | Welfare gap, static policy |
| `thr gap` | Welfare gap, threshold policy |
| `gain C` | Welfare improvement, centralized |
| `gain D` | Welfare improvement, decentralized |

To compute only some columns of the table, shorten `RATIOS` in the *Instance* cell.

**Case-study notebooks** (`Case Study DeepSWE/`). These use the GPT-5.6 Sol trials of the DeepSWE benchmark, with reasoning effort as the service depth.

- Each notebook first checks the data files against the checksums in `data/manifest.json`.
- Service time is a cost-based proxy, $S=\kappa C$. Here $C$ is a trial's recorded cost in USD, and $\kappa$ makes the mean service time over all 2,260 Sol trials equal to one. It is not measured wall-clock time.
- The delay sensitivities in Figures 6 to 9 are illustrative choices, stated in each notebook. They are not estimated from the data.
- `DeepSWE_stats.ipynb` can also show GPT-5.6 Luna and Terra: set `MODEL` in its first code cell. Figures are saved only for Sol.

## Case-study data

The DeepSWE trial data are not included in this repository. `prepare_data.py` downloads the public [DeepSWE v1.1 trial export](https://deepswe.datacurve.ai/artifacts/v1.1/trials.json) (about 51 MB) and writes two files to `Case Study DeepSWE/data/`:

| File | Contents |
| :--- | :--- |
| `deepswe_v1.1_gpt56_scored_trials.json` | The 6,766 GPT-5.6 Luna, Terra and Sol trials on DeepSWE tasks that the publisher includes in its score |
| `deepswe_v1.1_sol_excluded_trials.json` | The 4 GPT-5.6 Sol trials that the publisher excludes because the verifier timed out. The paper counts them as failures. |

Records are copied without changes. Before writing each file, the script checks its record count and SHA-256 checksum against `data/manifest.json`.

If the script cannot download the export, for example because you are behind a proxy, download it in a browser and pass its path:

```bash
python prepare_data.py --source path/to/trials.json
```

If the script reports a mismatch, the publisher has changed the records used here, and the paper's results cannot be reproduced exactly from the current export. The export used for the paper was last modified on 30 September 2026.

Use of the DeepSWE data is subject to its publisher's terms. The [DeepSWE release](https://deepswe.datacurve.ai/blog/deepswe) describes the benchmark and its methodology.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── Proposition_1.ipynb                         Figure 1
├── Section_6.1.ipynb                           Tables 1 and 2
├── Section_6.2.ipynb                           Figures 2 and 3
├── static_vs_threshold.ipynb                   Table 4
├── prop1_*.pdf, sqrt_*.pdf, log_*.pdf          figures saved by the notebooks above
└── Case Study DeepSWE/
    ├── prepare_data.py                         downloads the data (run this first)
    ├── DeepSWE_stats.ipynb                     Table 3, Figures 4 and 5
    ├── two_type_load.ipynb                     Figure 6
    ├── two_type_composition.ipynb              Figure 7
    ├── continuous_type_load.ipynb              Figure 8
    ├── continuous_type_personalization.ipynb   Figure 9
    ├── data/
    │   └── manifest.json                       data source, selection and checksums
    └── results/                                figures saved by the case-study notebooks
```

## License

The code is released under the MIT License; see [LICENSE](LICENSE). The license does not cover the DeepSWE data, which are not distributed here.
