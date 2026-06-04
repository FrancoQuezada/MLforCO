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

- No formal project name beyond the directory name was found.
- No automated tests were found.
- No dependency manifest was found.
- No command-line interface or argument parsing was found.
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
