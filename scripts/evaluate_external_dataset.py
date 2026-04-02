from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.modeling import get_or_train_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an outside molecule dataset against the toxicity pipeline.")
    parser.add_argument("input_csv", type=Path, help="CSV containing a smiles column and optional binary label column.")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name. Default: smiles")
    parser.add_argument("--label-col", default="", help="Optional binary label column for external evaluation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "external_eval",
        help="Directory for scored CSV and markdown report.",
    )
    return parser.parse_args()


def safe_metric(func, y_true: pd.Series, y_score: pd.Series) -> float | None:
    try:
        value = func(y_true, y_score)
    except Exception:
        return None
    return float(value)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column: {args.smiles_col}")

    pipeline = get_or_train_pipeline(force_retrain=False)
    rows: list[dict[str, object]] = []
    invalid_count = 0

    for row_index, row in df.iterrows():
        smiles = row[args.smiles_col]
        try:
            result = pipeline.predict_smiles(smiles)
        except Exception:
            invalid_count += 1
            rows.append(
                {
                    "row_index": int(row_index),
                    "smiles": smiles,
                    "valid_smiles": False,
                }
            )
            continue

        top_endpoint = result["predictions"].iloc[0]
        rows.append(
            {
                "row_index": int(row_index),
                "smiles": result["smiles"],
                "valid_smiles": True,
                "tox21_mean_risk": result["overall_risk"],
                "tox21_max_risk": result["max_endpoint_risk"],
                "composite_risk": result["composite_risk"],
                "tox21_band": result["tox21_risk_band"],
                "final_verdict": result["risk_band"],
                "confidence": result["confidence"],
                "ood_score": result["ood_score"],
                "applicability_domain": result["applicability_domain"],
                "structural_alert_score": result["structural_alert_score"],
                "structural_alert_band": result["structural_alert_band"],
                "structural_alerts": "; ".join(result["structural_alerts"]),
                "risk_reasons": "; ".join(result["risk_reasons"]),
                "top_endpoint": top_endpoint["target"],
                "top_endpoint_probability": float(top_endpoint["probability"]),
            }
        )

    scored = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "external_scored.csv"
    scored.to_csv(scored_path, index=False)

    valid = scored[scored["valid_smiles"] == True].copy()
    summary_lines = [
        "# External Evaluation Report",
        "",
        f"- Input rows: {len(df)}",
        f"- Valid SMILES: {len(valid)}",
        f"- Invalid SMILES: {invalid_count}",
    ]

    if not valid.empty:
        summary_lines.extend(
            [
                f"- Mean composite risk: {valid['composite_risk'].mean():.3f}",
                f"- Mean Tox21 mean risk: {valid['tox21_mean_risk'].mean():.3f}",
                f"- Mean OOD score: {valid['ood_score'].mean():.3f}",
                "",
                "## Final Verdict Counts",
                valid["final_verdict"].value_counts(dropna=False).to_markdown(),
                "",
                "## Applicability Counts",
                valid["applicability_domain"].value_counts(dropna=False).to_markdown(),
                "",
                "## Top Structural Alerts",
                valid["structural_alerts"].replace("", pd.NA).dropna().value_counts().head(10).to_markdown(),
            ]
        )

    if args.label_col:
        if args.label_col not in df.columns:
            raise ValueError(f"Missing label column: {args.label_col}")
        labeled = scored.join(df[[args.label_col]], how="left")
        labeled = labeled[(labeled["valid_smiles"] == True) & labeled[args.label_col].notna()].copy()
        if not labeled.empty:
            y_true = labeled[args.label_col].astype(int)
            y_comp = labeled["composite_risk"].astype(float)
            y_max = labeled["tox21_max_risk"].astype(float)
            y_pred = (y_comp >= 0.5).astype(int)
            summary_lines.extend(
                [
                    "",
                    "## External Label Metrics",
                    f"- ROC-AUC (composite): {safe_metric(roc_auc_score, y_true, y_comp)}",
                    f"- PR-AUC (composite): {safe_metric(average_precision_score, y_true, y_comp)}",
                    f"- ROC-AUC (max endpoint): {safe_metric(roc_auc_score, y_true, y_max)}",
                    f"- PR-AUC (max endpoint): {safe_metric(average_precision_score, y_true, y_max)}",
                    f"- Precision @ composite>=0.5: {safe_metric(precision_score, y_true, y_pred)}",
                    f"- Recall @ composite>=0.5: {safe_metric(recall_score, y_true, y_pred)}",
                ]
            )

            false_negatives = labeled[(y_true == 1) & (y_pred == 0)].head(10)
            if not false_negatives.empty:
                summary_lines.extend(
                    [
                        "",
                        "## Example False Negatives",
                        false_negatives[
                            [
                                "smiles",
                                "composite_risk",
                                "tox21_mean_risk",
                                "tox21_max_risk",
                                "ood_score",
                                "applicability_domain",
                                "structural_alerts",
                                "risk_reasons",
                            ]
                        ].to_markdown(index=False),
                    ]
                )

    report_path = args.output_dir / "external_report.md"
    report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Scored rows saved to: {scored_path}")
    print(f"Markdown report saved to: {report_path}")


if __name__ == "__main__":
    main()
