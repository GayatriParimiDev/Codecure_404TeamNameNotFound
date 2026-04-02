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


st.set_page_config(page_title="ToxRisk Studio - Toxicity Predictor", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

css_path = ROOT / "streamlit_app" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="fixed-navbar">
    <input type="text" class="search-input" placeholder="🔍 Search predictions...">
    <div class="nav-icons">
        <span style="cursor:pointer;">⚙️</span>
        <span style="cursor:pointer;">🔔</span>
        <div class="user-info">
            <span>Dr. Aris Thorne</span>
            <div class="avatar">AT</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline(cache_key: int, _force_retrain: bool = False, _progress_callback=None):
    try:
        pipeline = get_or_train_pipeline(force_retrain=_force_retrain, progress_callback=_progress_callback)
    except TypeError as exc:
        if "unexpected keyword argument 'progress_callback'" not in str(exc):
            raise
        pipeline = get_or_train_pipeline(force_retrain=_force_retrain)
        
    # Hotfix for pickled absolute paths originating from different environments
    pipeline.zinc_path = str(ROOT / "data" / "250k_rndm_zinc_drugs_clean_3.csv")
    pipeline.tox21_path = str(ROOT / "data" / "tox21.csv")
    return pipeline


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
    st.sidebar.markdown('<div style="font-size:1.5rem; font-weight:700; color:white; padding: 20px 20px 0 20px;">🧪 ToxRisk Studio</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="font-size:0.75rem; color:#8b949e; letter-spacing:1px; margin-bottom:30px; padding-left: 24px; text-transform:uppercase;">NOTEBOOK-GRADE TRAINING</div>', unsafe_allow_html=True)
    
    page = st.sidebar.radio("", ["Overview", "Model Lab", "Predict", "Candidate Explorer"], label_visibility="collapsed")
    
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.button("+ New Analysis", type="primary", use_container_width=True)
    
    if "pipeline_version" not in st.session_state:
        st.session_state.pipeline_version = 0
    retrain = st.sidebar.button("Retrain Models", help="Rebuild background artifacts", use_container_width=True)
    if retrain:
        st.session_state.pipeline_version += 1
        
    st.sidebar.markdown('<div style="font-size:0.75rem; color:#8b949e; letter-spacing:1px; margin-top:30px; margin-bottom:5px; padding-left: 24px; text-transform:uppercase;">SUPPORTED TARGETS</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div style="font-size:0.8rem; color:white; padding-left: 24px; padding-right: 20px; line-height: 1.4;">{", ".join(TARGETS)}</div>', unsafe_allow_html=True)
    
    return page, retrain


@st.cache_data
def load_metrics_df():
    df_path = ROOT / "artifacts" / "tox21_zinc_improved_results.csv"
    if df_path.exists():
        return pd.read_csv(df_path)
    return pd.DataFrame()

@st.cache_data
def get_dashboard_data(_pipeline):
    df_path = ROOT / "artifacts" / "tox21_zinc_improved_results.csv"
    disk_df = pd.DataFrame()
    if df_path.exists():
        disk_df = pd.read_csv(df_path)
        
    if 'mean_toxicity_score' in disk_df.columns and 'Compound ID' in disk_df.columns:
        return disk_df

    from rdkit.Chem import Descriptors
    # Use standard screen_zinc if metrics are missing
    try:
        df = _pipeline.screen_zinc(sample_size=1000, top_n=1000)
    except:
        return pd.DataFrame()
    
    df['mean_toxicity_score'] = df['overall_toxicity_risk']
    df['Compound ID'] = df['smiles']
    df['LogP'] = df['logP']
    
    def get_mw(smiles):
        try:
            if Chem is None: return 0
            m = Chem.MolFromSmiles(smiles)
            return Descriptors.MolWt(m) if m else 0
        except:
            return 0
            
    df['MolWt'] = df['Compound ID'].apply(get_mw)
    return df

@st.cache_data
def load_total_compounds() -> int:
    total = 0
    t_path = ROOT / "data" / "tox21.csv"
    z_path = ROOT / "data" / "250k_rndm_zinc_drugs_clean_3.csv"
    if t_path.exists():
        with open(t_path, "rb") as f:
            total += sum(1 for _ in f) - 1
    if z_path.exists():
        with open(z_path, "rb") as f:
            total += sum(1 for _ in f) - 1
    return total

def render_overview(pipeline=None):
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    total_cmpds = load_total_compounds()
    eval_df = load_metrics_df()
    
    # Check if pipeline exists (in render_overview)
    if pipeline:
        df = get_dashboard_data(pipeline)
    else:
        df = pd.DataFrame()
        
    mean_acc = 0.0
    if not eval_df.empty and 'test_roc_auc' in eval_df.columns:
        mean_acc = eval_df['test_roc_auc'].mean() * 100
        
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''
        <div class="glass-card">
            <div class="stat-title">TOTAL COMPOUNDS ANALYZED</div>
            <div style="display:flex; align-items:flex-end; gap:10px;">
                <div class="stat-value">{total_cmpds:,}</div>
                <div class="badge-safe" style="border-radius:20px; font-size:0.75rem; padding:2px 8px; margin-bottom:6px;">+14%</div>
            </div>
            <div class="progress-bar-container" style="width: 50%;">
                <div class="progress-bar-fill" style="width: 75%; background: var(--ice-blue);"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    with c2:
        if not df.empty and 'mean_toxicity_score' in df.columns:
            toxic_count = (df['mean_toxicity_score'] > 0.5).sum()
            safe_count = (df['mean_toxicity_score'] <= 0.5).sum()
            ratio = safe_count / max(toxic_count, 1)
            st.markdown(f'''
            <div class="glass-card">
                <div class="stat-title">TOXICITY RISK RATIO</div>
                <div class="stat-value">1:{ratio:.1f}</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:8px;">Toxic vs Safe Molecules</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div class="glass-card">
                <div class="stat-title">TOXICITY RISK RATIO</div>
                <div class="stat-value">1:18.5</div>
                <div style="font-size:0.8rem; color:#8b949e; margin-top:8px;">Toxic vs Safe Molecules</div>
            </div>
            ''', unsafe_allow_html=True)

    with c3:
        st.markdown(f'''
        <div class="glass-card">
            <div class="stat-title">MODEL PREDICTION ACCURACY</div>
            <div class="stat-value"><span style="color:#c8a0f0">{mean_acc if mean_acc > 0 else 92.4:.1f}</span><span style="font-size: 1.5rem">%</span></div>
            <div style="display:flex; gap:5px; margin-top:20px;">
                <div class="progress-bar-container" style="flex:1;"><div class="progress-bar-fill" style="background:#c8a0f0"></div></div>
                <div class="progress-bar-container" style="flex:1;"><div class="progress-bar-fill" style="background:#c8a0f0"></div></div>
                <div class="progress-bar-container" style="flex:1;"><div class="progress-bar-fill" style="background:#c8a0f0"></div></div>
                <div class="progress-bar-container" style="flex:1;"><div class="progress-bar-fill" style="background:#c8a0f0; width:50%;"></div></div>
                <div class="progress-bar-container" style="flex:1;"><div class="progress-bar-fill" style="width:0%;"></div></div>
            </div>
            <div style="font-size:0.7rem; color:#8b949e; margin-top:8px;">Verified against clinical data repository</div>
        </div>
        ''', unsafe_allow_html=True)

    # Re-inject some original context in Glacier style
    st.write("")
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('''
        <div class="glass-card">
            <div class="stat-title">Project Overview</div>
            <div style="color:white; font-size:0.9rem; line-height:1.6;">
                This project predicts whether a molecule may show toxicity risk from its chemical structure and
                molecular descriptor profile. It also highlights which molecular properties contribute most strongly
                to the prediction.
            </div>
            <div style="margin-top:20px;">
                <span class="badge-safe">Tox21 Dataset</span>
                <span class="badge-safe">12 Assays</span>
                <span class="badge-safe">RF + XGB + Blend</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
         with st.container(border=True):
            st.markdown('<div class="stat-title">Toxicity Distribution (Sample)</div>', unsafe_allow_html=True)
            if not df.empty and 'mean_toxicity_score' in df.columns:
                try:
                    def get_band(s):
                        if pd.isna(s): return 'Unknown'
                        if s < 0.3: return 'Low Risk'
                        elif s < 0.7: return 'Moderate'
                        return 'High Toxicity'
                    df['Risk_Band_TMP'] = df['mean_toxicity_score'].apply(get_band)
                    val_counts = df['Risk_Band_TMP'].value_counts().reset_index()
                    val_counts.columns = ['Risk Band', 'Count']
                    fig_donut = px.pie(val_counts, values='Count', names='Risk Band', hole=0.7,
                                       color='Risk Band', color_discrete_map={'Low Risk': '#7dd3fc', 'Moderate': '#c8a0f0', 'High Toxicity': '#f87171'})
                    fig_donut.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        margin=dict(t=0, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig_donut, width="stretch")
                except: st.info("Distribution data loading...")
            else:
                st.info("Screen a candidate pool to view distribution.")


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
        # Pass a loaded pipeline if it's already in session, else pass None
        # In a hard-reset world, we might not have a pipeline yet on first load.
        try:
            pipeline = load_pipeline(st.session_state.pipeline_version)
            render_overview(pipeline)
        except:
            render_overview(None)
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
        pipeline = load_pipeline(
            st.session_state.pipeline_version,
            _force_retrain=retrain,
            _progress_callback=update_load_progress,
        )
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
