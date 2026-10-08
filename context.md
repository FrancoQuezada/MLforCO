# Project Context

## Functional Overview

This repository implements a computational experiment pipeline for machine-learning-assisted combinatorial optimization. The active target problem is the Multi-item Capacitated Lot-Sizing Problem (MCLSP), formulated as a mixed-integer linear program with continuous production and inventory variables and binary setup variables.

The system generates synthetic MCLSP instances, solves them with Gurobi to obtain ground truth and relaxation features, trains supervised classifiers to imitate optimal binary setup decisions, and evaluates a matheuristic that fixes predicted binary variables before solving a restricted MILP.

## High-Level Architecture

The active architecture is a four-phase batch pipeline:

1. Instance generation.
2. Exact and relaxed optimization.
3. Supervised ML model training.
4. ML-guided matheuristic evaluation.

`config.py` provides shared parameters and paths. `functions.py` provides most implementation details, including MCLSP model construction, feasibility repair routines, model training wrappers, plotting helpers, and legacy experimental variants.

## Main Components

### `config.py`

Central source of configuration for Experiment 0:

- Training and test instance parameters.
- Capacity and setup cost factors.
- Selected instance IDs.
- Solver work/time limits.
- Output paths under `results_exp0`.
- Feature families and combinations.
- Classifier list.
- Evaluation threshold and variable-fixing strategy.

### Pipeline Scripts

- `main.py`: command-line orchestration with `generate`, `solve`, `train`, `evaluate`, and `all`; runs scripts from the repository root and stops on a failed phase.
- `01_generate_instances.py`: calls `crear_base_datos`, filters configured instance IDs, and writes instance/metadata Excel files.
- `02_solve_optimal.py`: calls `optimal`, `optimal_relaxed_binary`, and `optimal_root_node`; writes optimal variables, relaxation variables, and metrics.
- `03_train_model.py`: loads training variables, builds `binaria` from `y_pre`, prepares feature combinations, trains classifiers, and saves pickled models/scalers.
- `04_evaluate_matheuristic.py`: loads test variables and trained artifacts, predicts setup probabilities, injects prediction columns, and evaluates restricted MILPs through `optimal_predict_threshUP`.

### `functions.py`

The shared library includes:

- Instance generation and replacement functions.
- Gurobi formulations for exact MCLSP solves, relaxed binaries, root-node relaxations, worst-case solves, and multiple prediction/fixing variants.
- Feasibility repair routines, notably `vars_a_levantar`.
- ML wrappers for SVM, KNN, Random Forest, XGBoost, Logistic Regression, Decision Tree, Gradient Boosting, and Naive Bayes. The active `ML_CONFIG` uses `DT`, `RF`, `KNN`, `SVM`, and `NB`.
- Prediction column helpers.
- Plotting, feature generation, and correlation helpers.

Not all functions in `functions.py` are active in the current numbered pipeline.

## Data Flow

```text
config.py
  |
01_generate_instances.py
  -> results_exp0/data/{train,test}/instances_*.xlsx
  -> results_exp0/data/{train,test}/metadata_*.xlsx
  |
02_solve_optimal.py
  -> results_exp0/data/optimal/optimal_variables_*.xlsx
  -> results_exp0/data/optimal/optimal_metrics_*.xlsx
  |
03_train_model.py
  -> results_exp0/models/model_*.pkl
  -> results_exp0/scalers/scaler_*.pkl
  -> results_exp0/params/*_params.xlsx
  -> results_exp0/results/training_metrics_consolidated_*.xlsx
  |
04_evaluate_matheuristic.py
  -> results_exp0/results/final_evaluation_metrics_*.xlsx
```

The active generated data schema uses one row per `(Instancia, Periodo, Item)` combination.

## Technologies Used

- Python batch scripts.
- Pandas DataFrames and Excel files for data exchange.
- Gurobi/Gurobipy for MILP modeling and solution.
- Scikit-learn for classical supervised classifiers and scaling.
- XGBoost is imported and has a wrapper, although it is not included in the active `ML_CONFIG`.
- Matplotlib and Seaborn for plotting helpers.
- Pdfplumber for extracting text from project PDFs.

## Important Assumptions

- The working directory is the repository root when scripts are executed.
- A valid Gurobi installation and license are available.
- Upstream phase outputs exist before running downstream phases.
- File names are derived from `config.py`; changing config values changes expected artifact names.
- Training and validation splits should be made by instance to avoid leaking rows from the same optimization instance.
- The final-period inventory is fixed to zero in the MCLSP model.
- Setup variable `y[1, j]` is fixed to 1 for every item in the first period.
- Very small production values are post-processed by setting corresponding setup indicators to zero when recomputing objective values.

## Design Decisions Identified

- Experiment parameters are centralized in `config.py`.
- The pipeline uses numbered scripts to make phase order explicit.
- Generated artifacts include the experiment parameters in their filenames for reproducibility.
- Feature sets are grouped into original parameters (`A`), relaxed solution values (`B`), and root-node relaxation values (`B_hat`).
- Classifier outputs are probabilities for binary setup activation.
- The evaluation matheuristic fixes predicted zeros and ones according to `EVALUATION_CONFIG['zeroOrOne']`.
- Infeasible prediction fixing is repaired by solving an auxiliary optimization model that determines which binary predictions to lift.
- Hyperparameter selection in the ML wrappers uses instance-level leave-one-group-out validation and selects parameters using a coefficient-of-variation criterion over custom feasibility-aware scores.

## Domain Concepts

- MCLSP: Multi-item Capacitated Lot-Sizing Problem.
- MILP: Mixed-Integer Linear Programming.
- Setup variable: a binary variable indicating whether item `j` is produced in period `t`.
- Production variable `x`: continuous production amount.
- Inventory variable `s`: continuous stock amount.
- Capacity constraint: period-level production capacity shared by items.
- Predict-then-optimize: train an ML model first, then use predictions inside an optimization workflow.
- Matheuristic: a heuristic method that embeds mathematical programming.
- Feasibility-aware discrimination rule: a thresholding and repair mechanism intended to transform ML probabilities into feasible binary decisions.

## Missing or Unclear Information

- No automated tests were found.
- No dependency manifest was found.
- `main.py` supplies an argparse CLI, documented in `README.md`.
- No explicit environment variable contract was found.
- Expected runtime for the full pipeline cannot be determined from code alone.
- Whether all legacy functions in `functions.py` are still intended to be maintained cannot be determined.

## Architectural Risks and Technical Debt

- `functions.py` is a large shared file with active, legacy, and exploratory code mixed together.
- Several helper functions contain hardcoded historical paths or assumptions that are not aligned with the current `results_exp0` pipeline.
- The code relies heavily on Excel artifacts, which makes schema drift easy and runtime failures likely if columns are renamed.
- Missing dependency and test files make reproducibility harder.
- Gurobi log parsing and callbacks are embedded in several optimization variants.
- Training is coupled to optimization-based validation, making development feedback loops slow.

## Verified Training and Evaluation Details

- The active classifiers are Decision Tree, Random Forest, K-nearest Neighbors, probability-enabled SVC, and Gaussian Naive Bayes.
- The wrappers split unique `Instancia` IDs with `test_size=3/12` and seed 42, then apply `LeaveOneGroupOut` to the fitting instances. Scaling is fitted on each fold's fitting data and on the final fitting subset.
- `ParameterSampler` samples hyperparameter configurations with seed 42. Selection minimizes the nonzero coefficient of variation of feasibility-aware fold scores, with a first-candidate fallback; it does not maximize mean accuracy.
- `vars_a_levantar_train` performs two work-limited MILP solves per validation instance and scores setup agreement excluding period 1. Phase 3 persists repaired holdout accuracy, but drops the separately returned raw accuracy.
- Each classifier/feature combination is trained and saved separately. There is no automatic selection of one overall deployment model.
- Online repair keeps above-threshold setups active and can change predicted-zero setups to one. `vars_a_levantar` uses `SolutionLimit=1`, so minimum-change optimality is not certified.
- The root-node callback collects relaxation values at node zero and terminates on a non-root callback. The `tl` argument is not applied in that routine; callback extraction failure can trigger instance replacement.
- The `LogisticR` helper references undefined `train_instances` and is incomplete. XGBoost, gradient boosting, resampling, feature expansion, feature selection, warm starts, and hints are outside the active numbered pipeline.

## Metric Interpretation

- `optimal` and `optimal_predict_threshUP` use Gurobi `WorkLimit`; time-oriented configuration comments do not describe wall-clock limits. The continuous-relaxation routine and online auxiliary repair have no explicit time/work cap.
- Reference `time_optimal`, `time_relaxed`, and `time_rootnode` receive solver work units from their helpers. These columns must not be interpreted as elapsed seconds.
- Phase-4 `prediction_time` measures variable-fixing construction, not classifier inference; `time_hat` excludes feature generation, full model construction, and separately recorded repair duration.
- `gap_hat` is the restricted solver's MIP gap, not relative objective loss against the unrestricted reference. Root-node metrics mix an incumbent `objVal` with costs recomputed from relaxation values, so they are not a certified bound series.
- Raw and post-processed objectives differ because negligible production triggers setup removal.
- Reference callback summary values are zeros. ROC/AUC/threshold values returned by the active ML wrappers are zero placeholders, not measured diagnostics.
- Phase-4 `feasible` stores a repaired setup dictionary rather than a Boolean or a feasibility rate. Fixing counters do not identify repair-flip counts separately.
- Work-limited reference solves may supply incumbent labels without proving optimality. File names containing `optimal` do not by themselves certify solver status.

## Publication Review and Retained Artifacts

- The supplied image is published unchanged at `docs/images/imitation_learning_framework.png` and embedded in `README.md`.
- Nine retained Excel workbooks were inspected: instance and metadata tables, solution/relaxation tables, reference metric tables, and one parameter-selection record. The instance-variable tables have unique `(Instancia, Periodo, Item)` keys and no missing values.
- The retained snapshot contains 12 training instances with 8 items/60 periods and 30 test instances with 32 items/60 periods. Only the `DT`/`A` model and matching scaler are present; both pickle metadata records identify scikit-learn 1.4.2.
- Consolidated phase-3 and phase-4 metric files are absent. Default evaluation needs the remaining configured model/scaler pairs from phase 3.
- `.gitignore` excludes new experiment outputs, caches, environments, editor settings, solver logs, the local console transcript, credentials, and license files. Existing tracked research datasets, PDFs/text, and model/scaler artifacts are retained.
- Python syntax, CLI help, workbook schemas, pickle metadata, and the supplied image were checked during publication. The runtime import check failed because scikit-learn was absent; XGBoost and optional pdfplumber were also absent. Gurobi imported but license initialization failed with `License expired 2025-11-24`.
- All eight Python files passed in-memory compilation, and all eight README relative links resolved. The generator function was extracted from its source AST and exercised on two three-period/two-item instances; repeated generation with seed 17 produced identical DataFrames without writing experiment outputs. This was a focused helper check, not a numbered-phase run.
- Credential-pattern scans found no matches in the tracked working files, workbook XML, or Git history. These checks do not establish an exhaustive security audit.
- The numbered phases were not rerun, dependencies were not installed into the user's environment, and the existing generated research artifacts were not modified. Manuscript performance claims and roadmap experiments have not been independently reproduced by this review.
