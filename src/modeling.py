from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any
import warnings

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

try:
    from sklearn.exceptions import InconsistentVersionWarning
except ImportError:  # pragma: no cover
    InconsistentVersionWarning = UserWarning

from src.chemistry import (
    attach_feature_values,
    build_support_artifacts,
    build_support_features_for_query,
    classify_applicability_domain,
    combine_risk_signals,
    classify_confidence,
    classify_risk,
    compute_ood_score,
    detect_structural_alerts,
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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LEGACY_XGBOOST_PICKLE_WARNING = "If you are loading a serialized model"


RF_CONFIGS = [
    {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 700, "max_depth": 18, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 24, "min_samples_leaf": 2, "max_features": 0.35},
]

XGB_CONFIGS = [
    {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.80,
        "min_child_weight": 3,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 900,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_weight": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
    },
    {
        "n_estimators": 700,
        "max_depth": 8,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
]

BLEND_WEIGHT_GRID = np.linspace(0.1, 0.9, 9)


@dataclass
class TargetArtifact:
    selected_model: str
    threshold: float
    rf_model: Any
    xgb_model: Any
    scale_pos_weight: float
    blend_weights: tuple[float, float] = (0.5, 0.5)
    rf_config: dict[str, Any] | None = None
    xgb_config: dict[str, Any] | None = None


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
                rf_best = {
                    "model": rf,
                    "config": {"strategy": "most_frequent"},
                    "val_prob": predict_probabilities(rf, X_val),
                }
                rf_best["val_metrics"] = evaluate_binary(y_val, rf_best["val_prob"])
                xgb_best = {
                    "model": xgb,
                    "config": {"strategy": "most_frequent"},
                    "val_prob": predict_probabilities(xgb, X_val),
                }
                xgb_best["val_metrics"] = evaluate_binary(y_val, xgb_best["val_prob"])
            else:
                rf_best = None
                for rf_config in RF_CONFIGS:
                    rf_candidate = build_rf(rf_config)
                    rf_candidate.fit(X_train_fit, y_train_fit)
                    rf_val_prob = predict_probabilities(rf_candidate, X_val)
                    rf_candidate_bundle = {
                        "model": rf_candidate,
                        "config": dict(rf_config),
                        "val_prob": rf_val_prob,
                        "val_metrics": evaluate_binary(y_val, rf_val_prob),
                    }
                    if rf_best is None or compare_metric_bundle(
                        rf_candidate_bundle["val_metrics"], rf_best["val_metrics"]
                    ):
                        rf_best = rf_candidate_bundle

                xgb_best = None
                for xgb_config in XGB_CONFIGS:
                    xgb_candidate = build_xgb(xgb_config, scale_pos_weight)
                    xgb_candidate.fit(X_train_fit, y_train_fit)
                    xgb_val_prob = predict_probabilities(xgb_candidate, X_val)
                    xgb_candidate_bundle = {
                        "model": xgb_candidate,
                        "config": dict(xgb_config),
                        "val_prob": xgb_val_prob,
                        "val_metrics": evaluate_binary(y_val, xgb_val_prob),
                    }
                    if xgb_best is None or compare_metric_bundle(
                        xgb_candidate_bundle["val_metrics"], xgb_best["val_metrics"]
                    ):
                        xgb_best = xgb_candidate_bundle

            blend_best = optimize_blend(y_val, rf_best["val_prob"], xgb_best["val_prob"])

            candidates = {
                "random_forest": (rf_best["val_metrics"], rf_best["val_prob"]),
                "xgboost": (xgb_best["val_metrics"], xgb_best["val_prob"]),
                "blend": (blend_best["val_metrics"], blend_best["val_prob"]),
            }
            selected_name = max(
                candidates.keys(),
                key=lambda name: (safe_metric(candidates[name][0]["pr_auc"]), safe_metric(candidates[name][0]["roc_auc"])),
            )
            threshold = choose_threshold(y_val, candidates[selected_name][1], min_precision=MIN_PRECISION)["threshold"]

            rf_test_prob = predict_probabilities(rf_best["model"], X_test)
            xgb_test_prob = predict_probabilities(xgb_best["model"], X_test)

            # FIX: Use a single helper (_blend_probs) to compute the blended
            # probability, both here during training and later in predict_target.
            # Previously the training path inlined the formula independently,
            # meaning a future edit to one site would silently diverge from the
            # other.  Using the helper eliminates that coupling.
            blend_test_prob = _blend_probs(
                rf_test_prob, xgb_test_prob,
                blend_best["weight_rf"], blend_best["weight_xgb"],
            )
            test_prob_map = {
                "random_forest": rf_test_prob,
                "xgboost": xgb_test_prob,
                "blend": blend_test_prob,
            }
            test_metrics = evaluate_binary(y_test, test_prob_map[selected_name], threshold=threshold)

            models[target] = TargetArtifact(
                selected_model=selected_name,
                threshold=float(threshold),
                rf_model=rf_best["model"],
                xgb_model=xgb_best["model"],
                scale_pos_weight=float(scale_pos_weight),
                blend_weights=(float(blend_best["weight_rf"]), float(blend_best["weight_xgb"])),
                rf_config=rf_best["config"],
                xgb_config=xgb_best["config"],
            )
            results.append(
                {
                    "target": target,
                    "train_n": int(len(idx_train)),
                    "val_n": int(len(idx_val)),
                    "test_n": int(len(idx_test)),
                    "positive_rate_train": float(y_train.mean()),
                    "rf_val_roc_auc": rf_best["val_metrics"]["roc_auc"],
                    "rf_val_pr_auc": rf_best["val_metrics"]["pr_auc"],
                    "xgb_val_roc_auc": xgb_best["val_metrics"]["roc_auc"],
                    "xgb_val_pr_auc": xgb_best["val_metrics"]["pr_auc"],
                    "blend_val_roc_auc": blend_best["val_metrics"]["roc_auc"],
                    "blend_val_pr_auc": blend_best["val_metrics"]["pr_auc"],
                    "selected_model": selected_name,
                    "selected_threshold": float(threshold),
                    "blend_weight_rf": float(blend_best["weight_rf"]),
                    "blend_weight_xgb": float(blend_best["weight_xgb"]),
                    "test_roc_auc": test_metrics["roc_auc"],
                    "test_pr_auc": test_metrics["pr_auc"],
                    "test_f1": test_metrics["f1"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                }
            )

        results_df = (
            pd.DataFrame(results)
            .sort_values(["test_pr_auc", "test_roc_auc"], ascending=False, na_position="last")
            .reset_index(drop=True)
        )
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
        temp_path = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(self, temp_path)
        os.replace(temp_path, path)
        return path

    @classmethod
    def load(cls, path: Path | str = PIPELINE_PATH) -> "ToxicityPipeline":
        pipeline = _load_joblib_artifact(path)
        pipeline._normalize_runtime_models()
        return pipeline

    @classmethod
    def load_from_model_dir(
        cls,
        model_dir: Path | str,
        tox21_path: Path | str = TOX21_PATH,
        zinc_path: Path | str = ZINC_PATH,
        progress_callback=None,
    ) -> "ToxicityPipeline":
        model_dir = Path(model_dir)
        report_progress(progress_callback, 0.05, "Reading exported model metadata")
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

        report_progress(progress_callback, 0.12, "Loading Tox21 and ZINC datasets")
        tox21 = pd.read_csv(tox21_path)
        zinc = pd.read_csv(zinc_path)
        zinc["smiles"] = zinc["smiles"].astype(str).str.strip()
        for col in ["logP", "qed", "SAS"]:
            zinc[col] = pd.to_numeric(zinc[col], errors="coerce")
        zinc = zinc.dropna(subset=["smiles", "logP", "qed", "SAS"]).reset_index(drop=True)

        zinc_support_sample_size = int(metadata.get("zinc_support_sample_size", ZINC_SUPPORT_SAMPLE_SIZE))
        if zinc_support_sample_size < len(zinc):
            zinc_support = zinc.sample(zinc_support_sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
        else:
            zinc_support = zinc.copy()

        report_progress(progress_callback, 0.24, "Featurizing molecules for cache rebuild")
        tox21_valid, tox21_desc, tox21_fp = featurize_dataframe(tox21, include_fingerprints=True)
        zinc_valid, zinc_desc, _ = featurize_dataframe(zinc_support, include_fingerprints=False)
        report_progress(progress_callback, 0.48, "Building chemistry-space support artifacts")
        scaler, kmeans, nn, support_lookup, tox21_support = build_support_artifacts(
            tox21_desc=tox21_desc,
            zinc_valid=zinc_valid,
            zinc_desc=zinc_desc,
        )
        split_series = build_cluster_split(tox21_desc, scaler, kmeans)
        X = pd.concat([tox21_desc, tox21_support, tox21_fp], axis=1)[metadata["feature_columns"]]

        models: dict[str, TargetArtifact] = {}
        results_df = load_results_table(model_dir)
        model_meta = metadata["models"]
        total_targets = max(len(metadata["targets"]), 1)
        for index, target in enumerate(metadata["targets"], start=1):
            progress = 0.58 + 0.30 * (index / total_targets)
            report_progress(progress_callback, progress, f"Loading endpoint model {index}/{total_targets}: {target}")
            target_info = model_meta[target]
            rf_model = _load_joblib_artifact(model_dir / Path(target_info["rf_model_path"]).name)
            xgb_model = load_xgb_model(model_dir / Path(target_info["xgb_model_path"]).name)
            if hasattr(rf_model, "n_jobs"):
                rf_model.n_jobs = 1
            if hasattr(xgb_model, "n_jobs"):
                xgb_model.n_jobs = 1
            models[target] = TargetArtifact(
                selected_model=target_info["selected_model"],
                threshold=float(target_info["threshold"]),
                rf_model=rf_model,
                xgb_model=xgb_model,
                scale_pos_weight=float(target_info.get("scale_pos_weight", 1.0)),
                blend_weights=tuple(target_info.get("blend_weights", [0.5, 0.5])),
                rf_config=target_info.get("rf_config"),
                xgb_config=target_info.get("xgb_config"),
            )
        if results_df.empty:
            results_df = pd.DataFrame(
                [
                    {
                        "target": target,
                        "selected_model": model_meta[target]["selected_model"],
                        "selected_threshold": float(model_meta[target]["threshold"]),
                        "test_roc_auc": np.nan,
                        "test_pr_auc": np.nan,
                        "test_f1": np.nan,
                        "test_recall": np.nan,
                    }
                    for target in metadata["targets"]
                ]
            )

        pipeline = cls(
            targets=list(metadata["targets"]),
            feature_columns=list(metadata["feature_columns"]),
            descriptor_columns=list(metadata["descriptor_columns"]),
            support_feature_columns=list(metadata["support_feature_columns"]),
            # FIX: Added explicit na_position="last" to sort_values so that targets
            # whose metrics are NaN (the fallback path above) are always placed at
            # the bottom of the leaderboard rather than floating to arbitrary
            # positions due to undefined NaN comparison behaviour in pandas.
            results_df=results_df.sort_values(
                ["test_pr_auc", "test_roc_auc"], ascending=False, na_position="last"
            ).reset_index(drop=True),
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
        pipeline._normalize_runtime_models()
        report_progress(progress_callback, 0.92, "Pipeline object ready")
        return pipeline

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

    def _normalize_runtime_models(self) -> None:
        for artifact in self.models.values():
            if hasattr(artifact.rf_model, "n_jobs"):
                artifact.rf_model.n_jobs = 1
            if hasattr(artifact.xgb_model, "n_jobs"):
                artifact.xgb_model.n_jobs = 1

    def predict_smiles(self, smiles: str) -> dict[str, Any]:
        valid_df, desc_df, fp_df = featurize_dataframe(pd.DataFrame({"smiles": [smiles]}), include_fingerprints=True)
        if valid_df.empty:
            raise ValueError("Invalid SMILES string.")

        model_desc_df = align_feature_frame(desc_df, self.descriptor_columns)
        support_df = build_support_features_for_query(
            model_desc_df,
            self.scaler,
            self.kmeans,
            self.nn,
            self.support_lookup,
            descriptor_columns=self.descriptor_columns,
        )
        X_query = align_feature_frame(pd.concat([model_desc_df, support_df, fp_df], axis=1), self.feature_columns)

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
        structural_alerts = detect_structural_alerts(valid_df.iloc[0]["smiles"])
        tox21_mean_risk = float(overall["probability"].mean())
        tox21_max_risk = float(overall["probability"].max())
        structural_alert_score = float(structural_alerts["structural_alert_score"])
        ood_score = compute_ood_score(support_series.to_dict(), self.support_reference_stats)
        composite_risk, composite_band, risk_reasons = combine_risk_signals(
            tox21_mean_risk,
            tox21_max_risk,
            structural_alert_score,
            ood_score,
        )
        return {
            "smiles": valid_df.iloc[0]["smiles"],
            "descriptors": desc_df.iloc[0].round(4),
            "support_features": support_series,
            "predictions": overall,
            "overall_risk": tox21_mean_risk,
            "max_endpoint_risk": tox21_max_risk,
            "tox21_risk_band": classify_risk(tox21_mean_risk),
            "risk_band": composite_band,
            "composite_risk": composite_risk,
            "confidence": classify_confidence(support_series.to_dict(), self.support_reference_stats),
            "ood_score": ood_score,
            "applicability_domain": classify_applicability_domain(ood_score),
            "structural_alerts": structural_alerts["alerts"],
            "structural_alert_score": structural_alert_score,
            "structural_alert_band": structural_alerts["structural_alert_band"],
            "risk_reasons": risk_reasons,
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
        model_desc_df = align_feature_frame(zinc_desc, self.descriptor_columns)
        support_df = build_support_features_for_query(
            model_desc_df,
            self.scaler,
            self.kmeans,
            self.nn,
            self.support_lookup,
            descriptor_columns=self.descriptor_columns,
        )
        X_query = align_feature_frame(pd.concat([model_desc_df, support_df, zinc_fp], axis=1), self.feature_columns)

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
        # FIX: Delegate to _blend_probs so the blend formula is defined in
        # exactly one place.  The training path (train() above) now also calls
        # this helper, so both paths are guaranteed to stay in sync.
        return _blend_probs(rf_prob, xgb_prob, artifact.blend_weights[0], artifact.blend_weights[1])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _blend_probs(
    rf_prob: np.ndarray,
    xgb_prob: np.ndarray,
    weight_rf: float,
    weight_xgb: float,
) -> np.ndarray:
    """Return a weighted blend of RF and XGB probability arrays.

    Centralising the formula here ensures that the training evaluation path
    and the inference path in predict_target are always identical.
    """
    return weight_rf * rf_prob + weight_xgb * xgb_prob


def build_cluster_split(tox21_desc: pd.DataFrame, scaler, kmeans) -> pd.Series:
    tox21_scaled = scaler.transform(tox21_desc)
    tox21_clusters = pd.Series(kmeans.predict(tox21_scaled), index=tox21_desc.index, name="cluster")
    cluster_ids = sorted(tox21_clusters.unique())
    rng = np.random.default_rng(RANDOM_STATE)
    # NOTE: rng.shuffle() operates in-place and returns None (unlike
    # random.shuffle which also returns None, but numpy's Generator.shuffle
    # behaviour is easy to confuse with sorted/np.random.permutation which
    # return new arrays). Do NOT write `cluster_ids = rng.shuffle(cluster_ids)`.
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
        # No threshold met the min_precision constraint.  Fall back to 0.30,
        # which is more permissive (higher recall, lower precision) than the
        # default 0.50.  This is intentional for imbalanced toxicity targets
        # where false negatives are the primary concern, but callers should be
        # aware that the resulting predictions may have low precision.
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
    """Apply SMOTE oversampling, restricted to descriptor columns only.

    FIX: Morgan fingerprint columns (fp_0 … fp_N) are binary bit vectors.
    SMOTE interpolates between neighbours in feature space, producing
    fractional values for those bits in synthetic samples — the resampled
    fingerprints are therefore no longer valid binary fingerprints.

    To avoid this, SMOTE is now fitted on the non-fingerprint descriptor
    columns only.  The fingerprint block for each synthetic sample is copied
    from the nearest real neighbour (i.e. whichever real sample SMOTE chose as
    the base for that synthetic point) rather than being interpolated.  This is
    achieved by tracking the SMOTE sample indices and joining the original
    fingerprint rows back in after resampling.
    """
    if not USE_SMOTE:
        return X_train, y_train
    if SMOTE is None:
        raise ImportError("imbalanced-learn is required when USE_SMOTE=True")
    if y_train.nunique() < 2 or int(y_train.sum()) <= SMOTE_K_NEIGHBORS:
        return X_train, y_train

    fp_cols = [c for c in X_train.columns if c.startswith("fp_")]
    desc_cols = [c for c in X_train.columns if not c.startswith("fp_")]

    sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)

    if fp_cols:
        # Resample descriptors only.
        X_desc_res, y_res = sampler.fit_resample(X_train[desc_cols], y_train)
        n_original = len(X_train)
        n_synthetic = len(X_desc_res) - n_original

        # For synthetic rows, copy fingerprints from the nearest original
        # neighbour in descriptor space using the indices SMOTE recorded.
        # imblearn stores the source indices in sampler.sample_indices_.
        # Synthetic samples occupy positions [n_original:] in the output;
        # their source indices are sampler.sample_indices_[n_original:].
        if hasattr(sampler, "sample_indices_") and n_synthetic > 0:
            src_indices = sampler.sample_indices_[n_original:]
            fp_original = X_train[fp_cols].values
            fp_synthetic = fp_original[src_indices]
        else:
            # Fallback: repeat the mean fingerprint (all zeros for sparse fps).
            fp_synthetic = np.zeros((n_synthetic, len(fp_cols)), dtype=X_train[fp_cols].dtype)

        fp_original_block = X_train[fp_cols].values
        fp_block = np.vstack([fp_original_block, fp_synthetic])
        fp_df = pd.DataFrame(fp_block, columns=fp_cols)
        X_res = pd.concat([X_desc_res.reset_index(drop=True), fp_df], axis=1)[list(X_train.columns)]
    else:
        # No fingerprint columns present — resample the full feature matrix.
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    return X_res, y_res


def compare_metric_bundle(left: dict[str, float], right: dict[str, float]) -> bool:
    return (safe_metric(left["pr_auc"]), safe_metric(left["roc_auc"])) > (
        safe_metric(right["pr_auc"]),
        safe_metric(right["roc_auc"]),
    )


def optimize_blend(y_true: pd.Series, rf_prob: np.ndarray, xgb_prob: np.ndarray) -> dict[str, Any]:
    best = None
    for weight_rf in BLEND_WEIGHT_GRID:
        blend_prob = _blend_probs(rf_prob, xgb_prob, float(weight_rf), float(1.0 - weight_rf))
        candidate = {
            "weight_rf": float(weight_rf),
            "weight_xgb": float(1.0 - weight_rf),
            "val_prob": blend_prob,
            "val_metrics": evaluate_binary(y_true, blend_prob),
        }
        if best is None or compare_metric_bundle(candidate["val_metrics"], best["val_metrics"]):
            best = candidate
    return best


def load_xgb_model(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        if XGBClassifier is None:  # pragma: no cover
            raise ImportError("xgboost is required to load exported XGBoost models")
        model = XGBClassifier()
        model.load_model(path)
        return model
    return _load_joblib_artifact(path)


def _load_joblib_artifact(path: Path | str):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        warnings.filterwarnings("ignore", message=f".*{LEGACY_XGBOOST_PICKLE_WARNING}.*", category=UserWarning)
        return joblib.load(path)


def build_rf(config: dict[str, Any]):
    return RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        n_jobs=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_xgb(config: dict[str, Any], scale_pos_weight: float):
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError("xgboost is required to reproduce the improved notebook pipeline")
    return XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        min_child_weight=config["min_child_weight"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        scale_pos_weight=float(scale_pos_weight),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        verbosity=0,
        n_jobs=1,
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


def load_results_table(model_dir: Path) -> pd.DataFrame:
    candidates = [
        model_dir.parent / "tox21_zinc_improved_results.csv",
        model_dir.parent / "outputs_improved" / "tox21_zinc_improved_results.csv",
        model_dir / "tox21_zinc_improved_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


def align_feature_frame(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Feature frame is missing required columns: "
            + ", ".join(missing_columns[:10])
            + ("..." if len(missing_columns) > 10 else "")
        )
    return frame.reindex(columns=expected_columns)


def pipeline_cache_is_current(path: Path, model_dir: Path, metadata_path: Path) -> bool:
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        pipeline_mtime = path.stat().st_mtime
        metadata_mtime = metadata_path.stat().st_mtime
    except Exception:
        return False

    if metadata_mtime > pipeline_mtime:
        return False

    try:
        for artifact_path in model_dir.iterdir():
            if artifact_path.is_file() and artifact_path.stat().st_mtime > pipeline_mtime:
                return False
    except Exception:
        return False

    return True


def report_progress(progress_callback, fraction: float, message: str) -> None:
    fraction = max(0.0, min(1.0, float(fraction)))
    logger.info("Pipeline cache %.0f%%: %s", fraction * 100.0, message)
    if progress_callback is not None:
        progress_callback(fraction, message)


def get_or_train_pipeline(force_retrain: bool = False, path: Path | str = PIPELINE_PATH, progress_callback=None) -> ToxicityPipeline:
    path = Path(path)
    model_dir = path.parent / "models_improved"
    metadata_path = model_dir / "metadata.json"
    report_progress(progress_callback, 0.02, "Checking cached artifacts")
    if not force_retrain:
        if path.exists():
            if metadata_path.exists() and not pipeline_cache_is_current(path, model_dir, metadata_path):
                report_progress(progress_callback, 0.08, "Cached pipeline is outdated, rebuilding from exported models")
                pipeline = ToxicityPipeline.load_from_model_dir(model_dir, progress_callback=progress_callback)
                report_progress(progress_callback, 0.96, "Saving refreshed pipeline cache")
                pipeline.save(path)
                report_progress(progress_callback, 1.00, "Pipeline cache refreshed")
                return pipeline
            report_progress(progress_callback, 1.00, "Loaded cached pipeline")
            return ToxicityPipeline.load(path)
        if metadata_path.exists():
            report_progress(progress_callback, 0.08, "No pipeline cache found, building from exported models")
            pipeline = ToxicityPipeline.load_from_model_dir(model_dir, progress_callback=progress_callback)
            report_progress(progress_callback, 0.96, "Saving first pipeline cache")
            pipeline.save(path)
            report_progress(progress_callback, 1.00, "Pipeline cached")
            return pipeline
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_progress(progress_callback, 0.08, "Training a new pipeline locally")
    pipeline = ToxicityPipeline.train()
    report_progress(progress_callback, 0.96, "Saving trained pipeline cache")
    pipeline.save(path)
    report_progress(progress_callback, 1.00, "Pipeline cached")
    return pipeline
