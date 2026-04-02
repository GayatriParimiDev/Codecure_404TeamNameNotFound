# Project Rules

## 1. Scope Rules

- This project is for **Problem Statement 1: Drug Toxicity Prediction** only.
- `tox21.csv` is the only supervised training dataset in scope.
- `250k_rndm_zinc_drugs_clean_3.csv` is used for screening and ranking, not for supervised label learning.
- We will not add major optional datasets unless the core pipeline is already complete.

## 2. Modeling Rules

- Train **one model per toxicity endpoint**.
- Do not force a single joint model in the first version.
- Missing labels must be handled per target.
- Every prediction path must use the exact same feature pipeline as training.
- Start with classical ML baselines before trying complex architectures.
- `RandomForestClassifier` is the required baseline.
- `XGBClassifier` is the preferred boosted model if installation/runtime is stable.
- Prefer interpretable models or interpretable outputs.
- Use a cluster-based split with `KMeans` for train/validation/test assignment instead of a purely random split.
- Handle imbalance with class weighting first; apply SMOTE only on training folds if needed.
- Initial training implementation will be notebook-first for Colab.

## 3. Data Rules

- `smiles` is the canonical molecule input field.
- Invalid SMILES must be filtered and logged.
- Do not silently drop rows without tracking counts.
- ZINC SMILES must be stripped for newline and whitespace artifacts before use.
- No manual relabeling of toxicity endpoints.

## 4. Evaluation Rules

- Report metrics per target, not only one overall number.
- Use ROC-AUC and PR-AUC as primary metrics.
- Record class balance for each endpoint.
- Do not claim external validation using ZINC because it is unlabeled.
- Do not overclaim biological conclusions from heuristic ranking.

## 5. App Rules

- The app will be built in Streamlit.
- Inference will run server-side or locally in the app process.
- We will not build true mobile or browser-only on-device inference in this hackathon version.
- The app must accept a SMILES input and return endpoint-level toxicity predictions.
- The app must show an overall risk summary and at least one explanation artifact.

## 6. Engineering Rules

- Keep the code modular: data prep, features, training, prediction, app.
- Save trained models to disk and load them in the app.
- Do not duplicate feature logic across scripts.
- Every major output should be saved under `outputs/` or `models/`.
- Prefer reproducibility over premature optimization.

## 7. Time Management Rules

- Finish a working baseline before tuning.
- Finish one clean app page before adding advanced UI features.
- Finish core plots before presentation polish.
- If blocked, reduce scope instead of rewriting the whole approach.
- Do not switch to a GNN unless the RF/XGBoost pipeline is already complete and benchmarked.

## 8. Demo Rules

- The demo must work from a single user journey:
- input SMILES
- compute features
- predict 12 endpoint risks
- show aggregate risk
- show explanation
- optionally show safer screened candidates

- The demo should avoid long waits.
- Expensive scoring jobs must be precomputed where possible.

## 9. Claim Rules

- We can say the system predicts toxicity risk from molecular structure.
- We can say the ZINC set is used for candidate prioritization.
- We cannot say the model proves safety.
- We cannot say shortlisted molecules are experimentally validated.
- We cannot say unlabeled screening results are ground truth.

## 10. Repository Rules

- Planning docs stay in the repo root.
- Source code goes in `src/`.
- App entrypoint stays simple.
- Notebooks are allowed for exploration, but the final demo path must be script/app based.
