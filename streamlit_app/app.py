from __future__ import annotations

import sys
import time
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DEFAULT_SCREEN_SAMPLE_SIZE, TARGETS
from src.chemistry import mol_from_smiles
from src.modeling import get_or_train_pipeline

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:  # pragma: no cover
    Chem = None
    Draw = None


st.set_page_config(page_title="ToxRisk Studio", page_icon="T", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
        --bg: #f5efe4;
        --card: rgba(255, 252, 245, 0.88);
        --ink: #1f1a16;
        --muted: #66594f;
        --accent: #005f73;
        --accent-soft: #e0f0ee;
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(237, 205, 138, 0.35), transparent 28%),
            radial-gradient(circle at bottom right, rgba(0, 95, 115, 0.18), transparent 35%),
            linear-gradient(180deg, #f4eddc 0%, #efe5d2 100%);
        color: var(--ink);
    }
    .hero-card, .panel-card {
        background: var(--card);
        border: 1px solid rgba(31, 26, 22, 0.08);
        border-radius: 24px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 16px 42px rgba(31, 26, 22, 0.08);
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-family: Georgia, "Palatino Linotype", serif;
        font-size: 3rem;
        line-height: 1;
        margin-bottom: 0.35rem;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        max-width: 52rem;
    }
    .stat-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .stat-box {
        background: rgba(255, 255, 255, 0.62);
        border-radius: 18px;
        padding: 0.9rem 1rem;
    }
    .stat-label {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-value {
        font-size: 1.35rem;
        font-weight: 700;
    }
    .pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.82rem;
        margin-right: 0.45rem;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_pipeline(cache_key: int, force_retrain: bool, _progress_callback=None):
    try:
        return get_or_train_pipeline(force_retrain=force_retrain, progress_callback=_progress_callback)
    except TypeError as exc:
        if "unexpected keyword argument 'progress_callback'" not in str(exc):
            raise
        return get_or_train_pipeline(force_retrain=force_retrain)


def risk_color(risk_band: str) -> str:
    if risk_band == "Low":
        return "#1b7f5b"
    if risk_band == "Moderate":
        return "#c0841a"
    if risk_band == "Elevated":
        return "#b85042"
    return "#8d0801"


def render_molecule(smiles: str):
    if Chem is None or Draw is None:
        return
    mol = mol_from_smiles(smiles)
    if mol is None:
        return
    st.image(Draw.MolToImage(mol, size=(420, 260)), use_container_width=False)


def sidebar_controls():
    st.sidebar.title("ToxRisk Studio")
    st.sidebar.caption("Notebook-grade training and inference, inside Streamlit.")
    page = st.sidebar.radio("Workspace", ["Overview", "Model Lab", "Predict", "Candidate Explorer"])
    if "pipeline_version" not in st.session_state:
        st.session_state.pipeline_version = 0
    retrain = st.sidebar.button("Retrain Models")
    if retrain:
        st.session_state.pipeline_version += 1
    st.sidebar.markdown("Targets")
    st.sidebar.caption(", ".join(TARGETS))
    return page, retrain


def render_overview():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Drug Toxicity Prediction</div>
            <div class="hero-copy">
                This project predicts whether a molecule may show toxicity risk from its chemical structure and
                molecular descriptor profile. It also highlights which molecular properties contribute most strongly
                to the prediction and exposes the workflow through a simple screening interface.
            </div>
            <div class="stat-strip">
                <div class="stat-box"><div class="stat-label">Primary Dataset</div><div class="stat-value">Tox21</div></div>
                <div class="stat-box"><div class="stat-label">Assays</div><div class="stat-value">12</div></div>
                <div class="stat-box"><div class="stat-label">Model Stack</div><div class="stat-value">RF + XGB + Blend</div></div>
                <div class="stat-box"><div class="stat-label">Interface</div><div class="stat-value">Streamlit</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Simple View")
        st.markdown(
            """
            - We take molecular structure data and convert it into properties the model can understand.
            - The model estimates whether a molecule may trigger toxicity-related risk.
            - The app shows which properties seem to push the prediction higher or lower.
            - Users can test one molecule at a time or screen a larger candidate pool.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Technical View")
        st.markdown(
            """
            - We featurize molecules with RDKit descriptors, Morgan fingerprints, and ZINC-derived chemistry-space support features.
            - We train per-assay toxicity models on Tox21 and compare `random_forest`, `xgboost`, and a learned blend.
            - Validation selects thresholds per endpoint and the app reuses the same inference pipeline used in training.
            - The interface exposes endpoint predictions, molecular drivers, structural alerts, and ZINC candidate screening.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    task_col1, task_col2 = st.columns(2)
    with task_col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Expected Tasks")
        st.markdown(
            """
            - Process molecular descriptor datasets
            - Train ML models to predict toxicity
            - Identify key structural features linked to toxicity
            - Build a simple prediction interface or visualization
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with task_col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("How This App Matches")
        st.markdown(
            """
            - `Model Lab` shows training results and assay performance.
            - `Predict` scores user-provided SMILES and explains the prediction.
            - `Risk Drivers` and structural alerts interpret the toxicity signal.
            - `Candidate Explorer` screens a larger chemistry pool through the same model stack.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_model_lab(pipeline):
    st.title("Model Lab")
    st.caption("This view reproduces the second notebook's evaluation table and adds interpretable molecular drivers.")

    st.dataframe(
        pipeline.leaderboard().style.format(
            {
                "selected_threshold": "{:.2f}",
                "test_roc_auc": "{:.6f}",
                "test_pr_auc": "{:.6f}",
                "test_f1": "{:.6f}",
                "test_recall": "{:.6f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_col, driver_col = st.columns([1.2, 1])
    with chart_col:
        fig = px.scatter(
            pipeline.results_df,
            x="test_roc_auc",
            y="test_pr_auc",
            size="test_recall",
            color="selected_model",
            hover_name="target",
            title="Assay Performance Landscape",
            color_discrete_sequence=["#005f73", "#b85042", "#ca6702"],
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with driver_col:
        target = st.selectbox("Inspect Target Drivers", pipeline.targets, index=0)
        st.markdown(f'<span class="pill">{target}</span>', unsafe_allow_html=True)
        st.dataframe(
            pipeline.feature_importance_table(target=target).head(12).style.format({"importance": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Global Molecular Drivers")
    global_drivers = pipeline.aggregate_feature_importance(top_n=12)
    fig = px.bar(
        global_drivers.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#e9d8a6", "#005f73"],
    )
    fig.update_layout(height=420, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def render_predict(pipeline):
    st.title("Predict")
    st.caption("Paste a SMILES string and run the same in-app feature pipeline used during training.")

    smiles = st.text_area("SMILES", value="CCOc1ccc2nc(S(N)(=O)=O)sc2c1", height=120)
    if st.button("Run Toxicity Prediction", type="primary"):
        try:
            result = pipeline.predict_smiles(smiles)
        except ValueError as exc:
            st.error(str(exc))
            return

        left, right = st.columns([1.15, 0.85])
        with left:
            tox21_risk_band = result.get("tox21_risk_band", result.get("risk_band", "Low"))
            final_risk_band = result.get("risk_band", tox21_risk_band)
            structural_alert_score = float(result.get("structural_alert_score", 0.0))
            structural_alert_band = result.get("structural_alert_band", "None")
            structural_alerts = result.get("structural_alerts", [])
            risk_reasons = result.get("risk_reasons", [])
            applicability_domain = result.get("applicability_domain", "Unknown")
            ood_score = float(result.get("ood_score", 0.0))
            max_endpoint_risk = float(result.get("max_endpoint_risk", result.get("overall_risk", 0.0)))
            composite_risk = float(result.get("composite_risk", result.get("overall_risk", 0.0)))
            color = risk_color(final_risk_band)
            alert_color = "#b85042" if structural_alert_score >= 0.4 else "#c0841a" if structural_alert_score > 0 else "#1b7f5b"
            st.markdown(
                f"""
                <div class="hero-card">
                    <div class="stat-strip">
                        <div class="stat-box"><div class="stat-label">Tox21 Mean Risk</div><div class="stat-value">{result.get("overall_risk", 0.0):.3f}</div></div>
                        <div class="stat-box"><div class="stat-label">Tox21 Max Risk</div><div class="stat-value">{max_endpoint_risk:.3f}</div></div>
                        <div class="stat-box"><div class="stat-label">Composite Risk</div><div class="stat-value">{composite_risk:.3f}</div></div>
                        <div class="stat-box"><div class="stat-label">Tox21 Band</div><div class="stat-value" style="color:{risk_color(tox21_risk_band)}">{tox21_risk_band}</div></div>
                        <div class="stat-box"><div class="stat-label">Final Verdict</div><div class="stat-value" style="color:{color}">{final_risk_band}</div></div>
                        <div class="stat-box"><div class="stat-label">Confidence</div><div class="stat-value">{result.get("confidence", "Unknown")}</div></div>
                        <div class="stat-box"><div class="stat-label">Applicability</div><div class="stat-value">{applicability_domain}</div></div>
                        <div class="stat-box"><div class="stat-label">Structure Alert</div><div class="stat-value" style="color:{alert_color}">{structural_alert_band}</div></div>
                        <div class="stat-box"><div class="stat-label">Active Alerts</div><div class="stat-value">{int(result["predictions"]["predicted_toxic"].sum())}</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if structural_alerts:
                st.markdown(
                    " ".join(f'<span class="pill">{alert}</span>' for alert in structural_alerts),
                    unsafe_allow_html=True,
                )
            if risk_reasons:
                st.caption("Why this verdict: " + ", ".join(risk_reasons))
            st.caption(f"Out-of-distribution score: {ood_score:.2f}")
        with right:
            render_molecule(result["smiles"])

        pred_col, drivers_col = st.columns([1.3, 1])
        with pred_col:
            st.subheader("Endpoint Predictions")
            prediction_df = result["predictions"].copy()
            prediction_df["probability"] = prediction_df["probability"].round(4)
            prediction_df["threshold"] = prediction_df["threshold"].round(2)
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)
        with drivers_col:
            st.subheader("Risk Drivers")
            st.caption("Globally important non-fingerprint features with this molecule's values.")
            st.dataframe(
                result["local_drivers"].style.format({"importance": "{:.5f}", "value": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )

        desc_col, support_col = st.columns(2)
        with desc_col:
            st.subheader("Molecular Descriptors")
            st.dataframe(
                result["descriptors"].rename("value").reset_index().rename(columns={"index": "descriptor"}),
                use_container_width=True,
                hide_index=True,
            )
        with support_col:
            st.subheader("Chemistry-Space Support")
            st.dataframe(
                result["support_features"].rename("value").reset_index().rename(columns={"index": "feature"}),
                use_container_width=True,
                hide_index=True,
            )
            st.subheader("Structural Alert Score")
            st.metric("Alert severity", f'{float(result.get("structural_alert_score", 0.0)):.2f}')


def render_candidate_explorer(pipeline):
    st.title("Candidate Explorer")
    st.caption("Score a sample from the ZINC library with the trained toxicity models.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sample_size = st.slider("ZINC sample size", min_value=250, max_value=10000, value=DEFAULT_SCREEN_SAMPLE_SIZE, step=250)
    with col2:
        top_n = st.slider("Top candidates", min_value=10, max_value=100, value=25, step=5)
    with col3:
        sort_hint = st.selectbox("Ranking focus", ["Balanced", "Lowest toxicity"], index=0)

    if st.button("Screen Candidate Pool", type="primary"):
        screened = pipeline.screen_zinc(sample_size=sample_size, top_n=max(top_n, 100))
        if sort_hint == "Lowest toxicity":
            screened = screened.sort_values(["overall_toxicity_risk", "qed", "SAS"], ascending=[True, False, True])
        screened = screened.head(top_n).reset_index(drop=True)

        st.dataframe(
            screened.style.format(
                {
                    "logP": "{:.3f}",
                    "qed": "{:.3f}",
                    "SAS": "{:.3f}",
                    "overall_toxicity_risk": "{:.4f}",
                    "candidate_score": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.scatter(
            screened,
            x="qed",
            y="overall_toxicity_risk",
            size="candidate_score",
            color="SAS",
            hover_data=["smiles", "logP"],
            color_continuous_scale=["#e9d8a6", "#b85042"],
            title="Low-Toxicity vs Drug-Likeness",
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def main():
    page, retrain = sidebar_controls()
    if page == "Overview":
        render_overview()
        return

    load_status = st.empty()
    load_progress = st.progress(0, text="Preparing in-app training artifacts")
    load_started_at = time.perf_counter()

    def update_load_progress(fraction: float, message: str):
        elapsed = max(time.perf_counter() - load_started_at, 0.01)
        eta = max((elapsed / max(fraction, 0.01)) - elapsed, 0.0) if fraction < 1.0 else 0.0
        load_progress.progress(
            int(round(fraction * 100)),
            text=f"{message} | elapsed {elapsed:.1f}s | est. remaining {eta:.1f}s",
        )
        load_status.caption(f"Artifact cache progress: {int(round(fraction * 100))}%")

    try:
        pipeline = load_pipeline(st.session_state.pipeline_version, retrain, _progress_callback=update_load_progress)
    except Exception as exc:
        load_progress.empty()
        load_status.empty()
        st.error(str(exc))
        st.info("Install the packages in `requirements.txt`, then rerun `streamlit run streamlit_app/app.py`.")
        return

    total_load_time = time.perf_counter() - load_started_at
    load_progress.progress(100, text=f"Pipeline ready | total load time {total_load_time:.1f}s")
    load_status.success(f"Artifact cache ready in {total_load_time:.1f}s")

    if page == "Model Lab":
        render_model_lab(pipeline)
    elif page == "Predict":
        render_predict(pipeline)
    else:
        render_candidate_explorer(pipeline)


if __name__ == "__main__":
    main()
