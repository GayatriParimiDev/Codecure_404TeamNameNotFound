# Project Plan

## Goal

Build a hackathon-ready toxicity prediction system for **CodeCure Problem Statement 1** using:

- `tox21.csv` for supervised training
- `250k_rndm_zinc_drugs_clean_3.csv` for candidate screening
- a Streamlit app for demo and prediction

## Project Outcome

By the end of the build, the project should be able to:

1. train toxicity prediction models on Tox21
2. predict toxicity from a user-provided SMILES string
3. explain the prediction using feature importance
4. score ZINC molecules and rank lower-risk candidates
5. present everything in a clean Streamlit interface

## Build Phases

### Phase 1: Data understanding

Deliverables:

- inspect label coverage in `tox21.csv`
- inspect column quality in both datasets
- document null-handling strategy
- clean newline and formatting issues in ZINC SMILES

Success criteria:

- both datasets load cleanly
- invalid SMILES are logged
- target columns are clearly identified

### Phase 2: Feature pipeline

Deliverables:

- one shared feature generation pipeline for training and inference
- RDKit descriptor generation
- Morgan fingerprint generation
- feature schema saved and reused

Success criteria:

- same feature code is used in training and prediction
- no train/inference feature mismatch

### Phase 3: Model training

Deliverables:

- one binary classification model per toxicity target
- `RandomForestClassifier` baseline per target
- `XGBClassifier` main model per target
- notebook-based Colab training workflow
- saved metrics per target
- saved trained model artifacts

Success criteria:

- all 12 targets train successfully
- cluster-based split is applied before training
- imbalance handling is applied only on training data
- each model has reproducible evaluation outputs

### Phase 4: Evaluation and interpretation

Deliverables:

- ROC-AUC and PR-AUC per target
- class balance summary
- feature importance outputs
- summary charts for README and app

Success criteria:

- at least one clear interpretability artifact is produced
- evaluation is understandable to judges in under 1 minute

### Phase 5: ZINC screening

Deliverables:

- scored ZINC dataset
- aggregate toxicity score
- top ranked safer candidate list

Success criteria:

- precomputed shortlist is available for app display
- ranking logic is documented as heuristic

### Phase 6: Streamlit app

Deliverables:

- SMILES input page
- prediction results page
- explanation section
- candidate explorer section

Success criteria:

- app runs locally with one command
- demo path works without manual notebook steps

### Phase 7: Final packaging

Deliverables:

- final README
- screenshots
- cleaned repo structure
- requirements file
- Colab notebook checked into the repo

Success criteria:

- another person can clone and run it
- core demo works end to end

## Implementation Order

Work in this order only:

1. data cleaning
2. feature generation
3. baseline training
4. evaluation
5. model saving/loading
6. ZINC scoring
7. Streamlit app
8. README polish

## Daily Priority Rules

- do not build UI before the training pipeline works
- do not screen ZINC before the feature pipeline is stable
- do not optimize models before a baseline is saved
- do not spend time on deep learning unless the baseline is already complete

## Minimum Viable Product

The MVP includes:

- trained per-target toxicity models
- a single prediction workflow in Streamlit
- one evaluation chart
- one feature-importance chart
- one screened candidate shortlist

## Stretch Goals

- SHAP explanations
- molecule rendering in app
- candidate similarity search
- downloadable prediction report
