from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

from src.chemistry import (
    attach_feature_values,
    build_support_artifacts,
    build_support_features_for_query,
    classify_confidence,
    classify_risk,
    compute_candidate_score,
    featurize_dataframe,
    summarize_support_distribution,
)
from src.config import (
    ARTIFACT_DIR,
    DEFAULT_SCREEN_SAMPLE_SIZE,
    MIN_PRECISION,
    PIPELINE_PATH,
    RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
    TARGETS,
    TOX21_PATH,
    USE_SMOTE,
    ZINC_PATH,
    ZINC_SUPPORT_SAMPLE_SIZE,
)


@dataclass
class TargetArtifact:
    selected_model: str
    threshold: float
    rf_model: Any
    xgb_model: Any
    scale_pos_weight: float
    blend_weights: tuple[float, float] = (0.5, 0.5)


@dataclass
class ToxicityPipeline:
    targets: list[str]
    feature_columns: list[str]
    descriptor_columns: list[str]
    support_feature_columns: list[str]
    results_df: pd.DataFrame
    models: dict[str, TargetArtifact]
    scaler: Any
    kmeans: Any
    nn: Any
    support_lookup: pd.DataFrame
    support_reference_stats: dict[str, float]
    zinc_support_sample_size: int
    tox21_path: str
    zinc_path: str

    @classmethod
    def train(
        cls,
        tox21_path: Path | str = TOX21_PATH,
        zinc_path: Path | str = ZINC_PATH,
        zinc_support_sample_size: int = ZINC_SUPPORT_SAMPLE_SIZE,
    ) -> "ToxicityPipeline":
        tox21 = pd.read_csv(tox21_path)
        zinc = pd.read_csv(zinc_path)
        zinc["smiles"] = zinc["smiles"].astype(str).str.strip()
        for col in ["logP", "qed", "SAS"]:
            zinc[col] = pd.to_numeric(zinc[col], errors="coerce")
        zinc = zinc.dropna(subset=["smiles", "logP", "qed", "SAS"]).reset_index(drop=True)

        if zinc_support_sample_size < len(zinc):
            zinc_support = zinc.sample(zinc_support_sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        else:
            zinc_support = zinc.copy()

        tox21_valid, tox21_desc, tox21_fp = featurize_dataframe(tox21, include_fingerprints=True)
        zinc_valid, zinc_desc, _ = featurize_dataframe(zinc_support, include_fingerprints=False)

        scaler, kmeans, nn, support_lookup, tox21_support = build_support_artifacts(
            tox21_desc=tox21_desc,
            zinc_valid=zinc_valid,
            zinc_desc=zinc_desc,
        )
        X = pd.concat([tox21_desc, tox21_support, tox21_fp], axis=1)
        split_series = build_cluster_split(tox21_desc, scaler, kmeans)

        results = []
        models = {}
        for target in TARGETS:
            work_df = tox21_valid[[target]].join(split_series).dropna(subset=[target]).copy()
            work_df[target] = work_df[target].astype(int)

            idx_train = work_df.index[work_df["split"] == "train"]
            idx_val = work_df.index[work_df["split"] == "val"]
            idx_test = work_df.index[work_df["split"] == "test"]

            X_train = X.loc[idx_train]
            X_val = X.loc[idx_val]
            X_test = X.loc[idx_test]
            y_train = work_df.loc[idx_train, target]
            y_val = work_df.loc[idx_val, target]
            y_test = work_df.loc[idx_test, target]

            X_train_fit, y_train_fit = maybe_apply_smote(X_train, y_train)
            pos = max(int(y_train.sum()), 1)
            neg = max(int((y_train == 0).sum()), 1)
            scale_pos_weight = neg / pos

            if y_train_fit.nunique() < 2:
                rf = DummyClassifier(strategy="most_frequent")
                xgb = DummyClassifier(strategy="most_frequent")
                rf.fit(X_train_fit, y_train_fit)
                xgb.fit(X_train_fit, y_train_fit)
            else:
                rf = build_rf()
                xgb = build_xgb(scale_pos_weight)
                rf.fit(X_train_fit, y_train_fit)
                xgb.fit(X_train_fit, y_train_fit)

            rf_val_prob = predict_probabilities(rf, X_val)
            rf_val_metrics = evaluate_binary(y_val, rf_val_prob)
            xgb_val_prob = predict_probabilities(xgb, X_val)
            xgb_val_metrics = evaluate_binary(y_val, xgb_val_prob)

            blend_val_prob = 0.5 * rf_val_prob + 0.5 * xgb_val_prob
            blend_val_metrics = evaluate_binary(y_val, blend_val_prob)

            candidates = {
                "random_forest": (rf_val_metrics, rf_val_prob),
                "xgboost": (xgb_val_metrics, xgb_val_prob),
                "blend": (blend_val_metrics, blend_val_prob),
            }
            selected_name = max(
                candidates.keys(),
                key=lambda name: (safe_metric(candidates[name][0]["pr_auc"]), safe_metric(candidates[name][0]["roc_auc"])),
            )
            threshold = choose_threshold(y_val, candidates[selected_name][1], min_precision=MIN_PRECISION)["threshold"]

            rf_test_prob = predict_probabilities(rf, X_test)
            xgb_test_prob = predict_probabilities(xgb, X_test)
            blend_test_prob = 0.5 * rf_test_prob + 0.5 * xgb_test_prob
            test_prob_map = {
                "random_forest": rf_test_prob,
                "xgboost": xgb_test_prob,
                "blend": blend_test_prob,
            }
            test_metrics = evaluate_binary(y_test, test_prob_map[selected_name], threshold=threshold)

            models[target] = TargetArtifact(
                selected_model=selected_name,
                threshold=float(threshold),
                rf_model=rf,
                xgb_model=xgb,
                scale_pos_weight=float(scale_pos_weight),
            )
            results.append(
                {
                    "target": target,
                    "train_n": int(len(idx_train)),
                    "val_n": int(len(idx_val)),
                    "test_n": int(len(idx_test)),
                    "positive_rate_train": float(y_train.mean()),
                    "rf_val_roc_auc": rf_val_metrics["roc_auc"],
                    "rf_val_pr_auc": rf_val_metrics["pr_auc"],
                    "xgb_val_roc_auc": xgb_val_metrics["roc_auc"],
                    "xgb_val_pr_auc": xgb_val_metrics["pr_auc"],
                    "blend_val_roc_auc": blend_val_metrics["roc_auc"],
                    "blend_val_pr_auc": blend_val_metrics["pr_auc"],
                    "selected_model": selected_name,
                    "selected_threshold": float(threshold),
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_f1": test_metrics["f1"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                }
            )

        results_df = pd.DataFrame(results).sort_values(["test_pr_auc", "test_roc_auc"], ascending=False).reset_index(drop=True)
        return cls(
            targets=list(TARGETS),
            feature_columns=list(X.columns),
            descriptor_columns=list(tox21_desc.columns),
            support_feature_columns=list(tox21_support.columns),
            results_df=results_df,
            models=models,
            scaler=scaler,
            kmeans=kmeans,
            nn=nn,
            support_lookup=support_lookup,
            support_reference_stats=summarize_support_distribution(tox21_support),
            zinc_support_sample_size=int(len(zinc_valid)),
            tox21_path=str(Path(tox21_path)),
            zinc_path=str(Path(zinc_path)),
        )

    def save(self, path: Path | str = PIPELINE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str = PIPELINE_PATH) -> "ToxicityPipeline":
        return joblib.load(path)

    def leaderboard(self) -> pd.DataFrame:
        return self.results_df[
            [
                "target",
                "selected_model",
                "selected_threshold",
                "test_roc_auc",
                "test_pr_auc",
                "test_f1",
                "test_recall",
            ]
        ].copy()

    def predict_smiles(self, smiles: str) -> dict[str, Any]:
        valid_df, desc_df, fp_df = featurize_dataframe(pd.DataFrame({"smiles": [smiles]}), include_fingerprints=True)
        if valid_df.empty:
            raise ValueError("Invalid SMILES string.")

        support_df = build_support_features_for_query(desc_df, self.scaler, self.kmeans, self.nn, self.support_lookup)
        X_query = pd.concat([desc_df, support_df, fp_df], axis=1)[self.feature_columns]

        rows = []
        for target in self.targets:
            artifact = self.models[target]
            probability = float(self.predict_target(artifact, X_query)[0])
            rows.append(
                {
                    "target": target,
                    "selected_model": artifact.selected_model,
                    "threshold": artifact.threshold,
                    "probability": probability,
                    "predicted_toxic": probability >= artifact.threshold,
                }
            )

        overall = pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)
        support_series = support_df.iloc[0].round(4)
        return {
            "smiles": valid_df.iloc[0]["smiles"],
            "descriptors": desc_df.iloc[0].round(4),
            "support_features": support_series,
            "predictions": overall,
            "overall_risk": float(overall["probability"].mean()),
            "risk_band": classify_risk(float(overall["probability"].mean())),
            "confidence": classify_confidence(support_series.to_dict(), self.support_reference_stats),
            "local_drivers": attach_feature_values(self.aggregate_feature_importance(top_n=8), X_query.iloc[0]),
        }

    def screen_zinc(self, sample_size: int = DEFAULT_SCREEN_SAMPLE_SIZE, top_n: int = 25) -> pd.DataFrame:
        zinc = pd.read_csv(self.zinc_path)
        zinc["smiles"] = zinc["smiles"].astype(str).str.strip()
        for col in ["logP", "qed", "SAS"]:
            zinc[col] = pd.to_numeric(zinc[col], errors="coerce")
        zinc = zinc.dropna(subset=["smiles", "logP", "qed", "SAS"]).reset_index(drop=True)
        if sample_size < len(zinc):
            zinc = zinc.sample(sample_size, random_state=RANDOM_STATE).reset_index(drop=True)

        zinc_valid, zinc_desc, zinc_fp = featurize_dataframe(zinc, include_fingerprints=True)
        support_df = build_support_features_for_query(zinc_desc, self.scaler, self.kmeans, self.nn, self.support_lookup)
        X_query = pd.concat([zinc_desc, support_df, zinc_fp], axis=1)[self.feature_columns]

        scored = zinc_valid.copy()
        for target in self.targets:
            scored[target] = self.predict_target(self.models[target], X_query)

        scored["overall_toxicity_risk"] = scored[self.targets].mean(axis=1)
        scored["candidate_score"] = compute_candidate_score(scored)
        keep = ["smiles", "logP", "qed", "SAS", "overall_toxicity_risk", "candidate_score", *self.targets]
        return scored[keep].sort_values(
            ["candidate_score", "overall_toxicity_risk", "qed"],
            ascending=[False, True, False],
        ).head(top_n).reset_index(drop=True)

    def feature_importance_table(self, target: str, include_fingerprints: bool = False) -> pd.DataFrame:
        artifact = self.models[target]
        rf_imp = model_feature_importance(artifact.rf_model, self.feature_columns)
        xgb_imp = model_feature_importance(artifact.xgb_model, self.feature_columns)
        if artifact.selected_model == "random_forest":
            scores = rf_imp
        elif artifact.selected_model == "xgboost":
            scores = xgb_imp
        else:
            scores = 0.5 * rf_imp + 0.5 * xgb_imp

        table = scores.rename("importance").reset_index().rename(columns={"index": "feature"})
        table = table.sort_values("importance", ascending=False).reset_index(drop=True)
        if not include_fingerprints:
            table = table[~table["feature"].str.startswith("fp_")].reset_index(drop=True)
        return table

    def aggregate_feature_importance(self, top_n: int = 12) -> pd.DataFrame:
        tables = []
        for target in self.targets:
            table = self.feature_importance_table(target=target, include_fingerprints=False).copy()
            total = table["importance"].sum()
            table["normalized_importance"] = table["importance"] / total if total else 0.0
            tables.append(table[["feature", "normalized_importance"]])

        merged = pd.concat(tables, axis=0)
        return (
            merged.groupby("feature", as_index=False)["normalized_importance"]
            .mean()
            .sort_values("normalized_importance", ascending=False)
            .head(top_n)
            .rename(columns={"normalized_importance": "importance"})
            .reset_index(drop=True)
        )

    def predict_target(self, artifact: TargetArtifact, X_query: pd.DataFrame) -> np.ndarray:
        rf_prob = predict_probabilities(artifact.rf_model, X_query)
        xgb_prob = predict_probabilities(artifact.xgb_model, X_query)
        if artifact.selected_model == "random_forest":
            return rf_prob
        if artifact.selected_model == "xgboost":
            return xgb_prob
        return artifact.blend_weights[0] * rf_prob + artifact.blend_weights[1] * xgb_prob


def build_cluster_split(tox21_desc: pd.DataFrame, scaler, kmeans) -> pd.Series:
    tox21_scaled = scaler.transform(tox21_desc)
    tox21_clusters = pd.Series(kmeans.predict(tox21_scaled), index=tox21_desc.index, name="cluster")
    cluster_ids = sorted(tox21_clusters.unique())
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(cluster_ids)
    train_cut = int(0.70 * len(cluster_ids))
    val_cut = int(0.85 * len(cluster_ids))
    train_clusters = set(cluster_ids[:train_cut])
    val_clusters = set(cluster_ids[train_cut:val_cut])
    return tox21_clusters.map(
        lambda cluster: "train" if cluster in train_clusters else ("val" if cluster in val_clusters else "test")
    ).rename("split")


def evaluate_binary(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    unique = len(np.unique(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if unique > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if unique > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray, min_precision: float = MIN_PRECISION) -> dict[str, float]:
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if precision >= min_precision and f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    if best["f1"] < 0:
        fallback = 0.30
        y_pred = (y_prob >= fallback).astype(int)
        best = {
            "threshold": float(fallback),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        }
    return best


def maybe_apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    if not USE_SMOTE:
        return X_train, y_train
    if SMOTE is None:
        raise ImportError("imbalanced-learn is required when USE_SMOTE=True")
    if y_train.nunique() < 2 or int(y_train.sum()) <= SMOTE_K_NEIGHBORS:
        return X_train, y_train
    sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
    return sampler.fit_resample(X_train, y_train)


def build_rf():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_xgb(scale_pos_weight: float):
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError("xgboost is required to reproduce the improved notebook pipeline")
    return XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=float(scale_pos_weight),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        verbosity=0,
    )


def predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 1:
            classes = getattr(model, "classes_", np.array([0]))
            return np.ones(len(X)) if int(classes[0]) == 1 else np.zeros(len(X))
        return probabilities[:, 1]
    return np.asarray(model.predict(X), dtype=float)


def model_feature_importance(model, feature_columns: list[str]) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_columns)
    return pd.Series(np.zeros(len(feature_columns), dtype=float), index=feature_columns)


def safe_metric(value: float) -> float:
    return -1.0 if pd.isna(value) else float(value)


def get_or_train_pipeline(force_retrain: bool = False, path: Path | str = PIPELINE_PATH) -> ToxicityPipeline:
    path = Path(path)
    if path.exists() and not force_retrain:
        return ToxicityPipeline.load(path)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = ToxicityPipeline.train()
    pipeline.save(path)
    return pipeline
