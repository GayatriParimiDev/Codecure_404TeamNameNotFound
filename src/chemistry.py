from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import FP_BITS, KMEANS_CLUSTERS, NN_NEIGHBORS, RANDOM_STATE


RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


STRUCTURAL_ALERT_PATTERNS: list[dict[str, object]] = [
    {
        "name": "Sulfonamide",
        "smarts": "[SX4](=O)(=O)[NX3;H0,H1,H2]",
        "severity": 0.35,
    },
    {
        "name": "Aromatic sulfonamide",
        "smarts": "[c,n][SX4](=O)(=O)[NX3;H0,H1,H2]",
        "severity": 0.70,
    },
    {
        "name": "Heteroaromatic sulfonamide",
        "smarts": "[n][SX4](=O)(=O)[NX3;H0,H1,H2]",
        "severity": 0.90,
    },
    {
        "name": "Aromatic nitro group",
        "smarts": "[c][N+](=O)[O-]",
        "severity": 0.85,
    },
    {
        "name": "Aniline-like aromatic amine",
        "smarts": "[c][NX3;H1,H2]",
        "severity": 0.60,
    },
    {
        "name": "Michael acceptor",
        "smarts": "[C,c]=[C,c][C](=O)[O,N,S,#6]",
        "severity": 0.80,
    },
    {
        "name": "Aldehyde",
        "smarts": "[CX3H1](=O)[#6]",
        "severity": 0.75,
    },
    {
        "name": "Epoxide",
        "smarts": "[OX2r3]1[#6r3][#6r3]1",
        "severity": 0.80,
    },
    {
        "name": "Hydrazine or hydrazone motif",
        "smarts": "[NX3,NX2][NX3,NX2]",
        "severity": 0.85,
    },
    {
        "name": "Isocyanate",
        "smarts": "[NX2]=[CX2]=[OX1]",
        "severity": 0.95,
    },
    {
        "name": "Acyl halide",
        "smarts": "[CX3](=O)[F,Cl,Br,I]",
        "severity": 0.95,
    },
]

# Per-alert co-occurrence penalty weight. Each additional matched alert beyond
# the first contributes this fraction of its own severity on top of the max.
# Capped at 1.0. Tune this constant to adjust sensitivity to alert accumulation.
_COOCCURRENCE_PENALTY_WEIGHT: float = 0.25


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None

    text = str(smiles).strip()
    if not text:
        return None

    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
        return mol
    except Chem.KekulizeException:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
            return mol
        except Exception:
            return None
    except Exception:
        return None


def morgan_fp(mol, radius: int = 2, n_bits: int = FP_BITS) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_descriptor_dict(mol) -> dict[str, float]:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "MolMR": Descriptors.MolMR(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "AromaticRings": Lipinski.NumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "HeteroAtoms": Lipinski.NumHeteroatoms(mol),
        "ValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NHOHCount": Lipinski.NHOHCount(mol),
        "NOCount": Lipinski.NOCount(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),
        "NumBridgeheadAtoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "NumSpiroAtoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "HallKierAlpha": Descriptors.HallKierAlpha(mol),
        "BalabanJ": Descriptors.BalabanJ(mol),
        "QED": QED.qed(mol),
    }


def featurize_dataframe(df: pd.DataFrame, include_fingerprints: bool = True):
    valid_idx = []
    desc_rows = []
    fp_rows = []

    for idx, smiles in df["smiles"].items():
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        valid_idx.append(idx)
        desc_rows.append(calc_descriptor_dict(mol))
        if include_fingerprints:
            fp_rows.append(morgan_fp(mol))

    valid_df = df.loc[valid_idx].copy()
    desc_df = pd.DataFrame(desc_rows, index=valid_idx)
    if include_fingerprints:
        fp_df = pd.DataFrame(fp_rows, index=valid_idx, columns=[f"fp_{i}" for i in range(FP_BITS)])
        return valid_df, desc_df, fp_df
    return valid_df, desc_df, None


def build_support_artifacts(
    tox21_desc: pd.DataFrame,
    zinc_valid: pd.DataFrame,
    zinc_desc: pd.DataFrame,
):
    descriptor_columns = list(tox21_desc.columns)
    scaler = StandardScaler()
    combined_space = pd.concat([tox21_desc[descriptor_columns], zinc_desc[descriptor_columns]], axis=0)
    combined_scaled = scaler.fit_transform(combined_space)

    kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(combined_scaled)

    zinc_scaled = scaler.transform(zinc_desc[descriptor_columns])
    support_lookup = zinc_valid[["logP", "qed", "SAS"]].reset_index(drop=True)
    nn = NearestNeighbors(n_neighbors=NN_NEIGHBORS, metric="euclidean")
    nn.fit(zinc_scaled)

    tox21_support = build_support_features_for_query(
        query_desc=tox21_desc,
        scaler=scaler,
        kmeans=kmeans,
        nn=nn,
        support_lookup=support_lookup,
    )
    return scaler, kmeans, nn, support_lookup, tox21_support


def build_support_features_for_query(
    query_desc: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    nn: NearestNeighbors,
    support_lookup: pd.DataFrame,
    descriptor_columns: list[str] | None = None,
) -> pd.DataFrame:
    if descriptor_columns is None:
        descriptor_columns = list(getattr(scaler, "feature_names_in_", query_desc.columns))
    aligned_query_desc = query_desc.reindex(columns=descriptor_columns)
    missing_columns = [column for column in descriptor_columns if column not in query_desc.columns]
    if missing_columns:
        raise ValueError(
            "Missing required descriptor columns for support-feature generation: "
            + ", ".join(missing_columns[:10])
            + ("..." if len(missing_columns) > 10 else "")
        )

    query_scaled = scaler.transform(aligned_query_desc)
    distances, neighbors = nn.kneighbors(query_scaled)

    neighbor_logp = support_lookup.iloc[neighbors.flatten()]["logP"].to_numpy().reshape(len(neighbors), NN_NEIGHBORS)
    neighbor_qed = support_lookup.iloc[neighbors.flatten()]["qed"].to_numpy().reshape(len(neighbors), NN_NEIGHBORS)
    neighbor_sas = support_lookup.iloc[neighbors.flatten()]["SAS"].to_numpy().reshape(len(neighbors), NN_NEIGHBORS)
    cluster_distance = kmeans.transform(query_scaled).min(axis=1)

    return pd.DataFrame(
        {
            "zinc_nn_dist_min": distances.min(axis=1),
            "zinc_nn_dist_mean": distances.mean(axis=1),
            "zinc_nn_dist_std": distances.std(axis=1),
            "zinc_local_density": 1.0 / (1.0 + distances.mean(axis=1)),
            "zinc_nn_logP_mean": neighbor_logp.mean(axis=1),
            "zinc_nn_logP_std": neighbor_logp.std(axis=1),
            "zinc_nn_qed_mean": neighbor_qed.mean(axis=1),
            "zinc_nn_qed_std": neighbor_qed.std(axis=1),
            "zinc_nn_SAS_mean": neighbor_sas.mean(axis=1),
            "zinc_nn_SAS_std": neighbor_sas.std(axis=1),
            "cluster_distance": cluster_distance,
        },
        index=query_desc.index,
    )


def summarize_support_distribution(tox21_support: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for column in ["zinc_nn_dist_mean", "cluster_distance"]:
        metrics[f"{column}_p50"] = float(tox21_support[column].quantile(0.50))
        metrics[f"{column}_p85"] = float(tox21_support[column].quantile(0.85))
        metrics[f"{column}_p95"] = float(tox21_support[column].quantile(0.95))
    return metrics


def classify_risk(overall_risk: float) -> str:
    if overall_risk < 0.25:
        return "Low"
    if overall_risk < 0.50:
        return "Moderate"
    if overall_risk < 0.75:
        return "Elevated"
    return "High"


def combine_risk_signals(
    overall_risk: float,
    max_endpoint_risk: float,
    structural_alert_score: float,
    ood_score: float = 0.0,
) -> tuple[float, str, list[str]]:
    # Clamp all inputs to [0, 1] defensively before any arithmetic.
    overall_risk = float(np.clip(overall_risk, 0.0, 1.0))
    max_endpoint_risk = float(np.clip(max_endpoint_risk, 0.0, 1.0))
    structural_alert_score = float(np.clip(structural_alert_score, 0.0, 1.0))
    ood_score = float(np.clip(ood_score, 0.0, 1.0))

    composite_risk = max(
        overall_risk,
        structural_alert_score,
        min(1.0, 0.85 * max_endpoint_risk),
    )
    reasons: list[str] = []

    if structural_alert_score >= 0.90:
        composite_risk = max(composite_risk, 0.95)
        reasons.append("high-severity structural alert")
    elif structural_alert_score >= 0.75:
        composite_risk = max(composite_risk, 0.85)
        reasons.append("strong structural alert")
    elif structural_alert_score >= 0.55:
        composite_risk = max(composite_risk, 0.65)
        reasons.append("moderate structural alert")

    if max_endpoint_risk >= 0.90:
        composite_risk = max(composite_risk, 0.95)
        reasons.append("very high endpoint probability")
    elif max_endpoint_risk >= 0.75:
        composite_risk = max(composite_risk, 0.80)
        reasons.append("high endpoint probability")
    elif max_endpoint_risk >= 0.55:
        composite_risk = max(composite_risk, 0.62)
        reasons.append("moderate endpoint probability")

    # FIX: Raised co-occurrence thresholds from (0.60, 0.55) to (0.75, 0.65) so
    # that a "High" composite score (>=0.90) is only triggered when both signals
    # are individually strong, consistent with the single-signal band thresholds.
    if structural_alert_score >= 0.75 and max_endpoint_risk >= 0.65:
        composite_risk = max(composite_risk, 0.90)
        reasons.append("structural alert plus endpoint agreement")

    if ood_score >= 0.85:
        composite_risk = max(composite_risk, 0.55)
        reasons.append("outside training chemistry space")
    elif ood_score >= 0.60:
        composite_risk = max(composite_risk, 0.35)
        reasons.append("edge of training chemistry space")

    # FIX: Clamp composite_risk to [0, 1] before returning.  Without this, an
    # overall_risk or structural_alert_score slightly above 1.0 (possible if a
    # caller passes a raw model output before clipping) would propagate through
    # the max() chain and produce an out-of-range composite score.
    composite_risk = float(np.clip(composite_risk, 0.0, 1.0))

    return composite_risk, classify_risk(composite_risk), reasons


def classify_confidence(support_features: dict[str, float], reference_stats: dict[str, float]) -> str:
    ood_score = compute_ood_score(support_features, reference_stats)
    if ood_score <= 0.20:
        return "High"
    if ood_score <= 0.60:
        return "Moderate"
    return "Low"


def compute_ood_score(support_features: dict[str, float], reference_stats: dict[str, float]) -> float:
    dist_mean = float(support_features["zinc_nn_dist_mean"])
    cluster_distance = float(support_features["cluster_distance"])

    def normalized_exceedance(value: float, p50: float, p85: float, p95: float) -> float:
        if value <= p50:
            return 0.0
        if value <= p85:
            return 0.5 * (value - p50) / max(p85 - p50, 1e-9)
        return min(1.0, 0.5 + 0.5 * (value - p85) / max(p95 - p85, 1e-9))

    dist_p50 = float(reference_stats["zinc_nn_dist_mean_p50"])
    dist_p85 = float(reference_stats["zinc_nn_dist_mean_p85"])
    dist_p95 = float(reference_stats.get("zinc_nn_dist_mean_p95", dist_p85 + max(dist_p85 - dist_p50, 1e-9)))
    cluster_p50 = float(reference_stats["cluster_distance_p50"])
    cluster_p85 = float(reference_stats["cluster_distance_p85"])
    cluster_p95 = float(reference_stats.get("cluster_distance_p95", cluster_p85 + max(cluster_p85 - cluster_p50, 1e-9)))

    dist_component = normalized_exceedance(
        dist_mean,
        dist_p50,
        dist_p85,
        dist_p95,
    )
    cluster_component = normalized_exceedance(
        cluster_distance,
        cluster_p50,
        cluster_p85,
        cluster_p95,
    )
    return float(np.clip(0.55 * dist_component + 0.45 * cluster_component, 0.0, 1.0))


def classify_applicability_domain(ood_score: float) -> str:
    if ood_score <= 0.20:
        return "In-domain"
    if ood_score <= 0.60:
        return "Edge"
    return "Out-of-domain"


def classify_structural_alert(severity: float) -> str:
    if severity <= 0:
        return "None"
    if severity < 0.4:
        return "Mild"
    if severity < 0.8:
        return "Moderate"
    return "High"


def detect_structural_alerts(mol_or_smiles) -> dict[str, object]:
    mol = mol_or_smiles if hasattr(mol_or_smiles, "HasSubstructMatch") else mol_from_smiles(mol_or_smiles)
    if mol is None:
        return {
            "alerts": [],
            "structural_alert_score": 0.0,
            "structural_alert_band": "None",
        }

    matched: list[str] = []
    max_severity: float = 0.0

    for pattern in STRUCTURAL_ALERT_PATTERNS:
        query = Chem.MolFromSmarts(str(pattern["smarts"]))
        if query is None:
            continue
        if mol.HasSubstructMatch(query):
            matched.append(str(pattern["name"]))
            max_severity = max(max_severity, float(pattern["severity"]))

    # FIX: Severity is no longer pure-max across matched alerts.  Each alert
    # beyond the first adds a co-occurrence penalty proportional to its own
    # severity, capped at 1.0.  This means a molecule matching three moderate
    # alerts scores higher than one matching only the worst single alert,
    # reflecting real accumulation of structural liability.
    #
    # Formula: score = min(1.0, max_severity + penalty_weight * sum(s_i for all
    # but the worst alert)).  The _COOCCURRENCE_PENALTY_WEIGHT constant (0.25)
    # can be tuned without changing the logic here.
    severity: float = max_severity
    if len(matched) > 1:
        all_severities = sorted(
            (float(p["severity"]) for p in STRUCTURAL_ALERT_PATTERNS if str(p["name"]) in matched),
            reverse=True,
        )
        # Sum all but the highest (which is already max_severity).
        co_penalty = _COOCCURRENCE_PENALTY_WEIGHT * sum(all_severities[1:])
        severity = min(1.0, max_severity + co_penalty)

    return {
        "alerts": matched,
        "structural_alert_score": severity,
        "structural_alert_band": classify_structural_alert(severity),
    }


def attach_feature_values(driver_table: pd.DataFrame, feature_row: pd.Series) -> pd.DataFrame:
    table = driver_table.copy()
    table["value"] = table["feature"].map(feature_row.to_dict())
    return table


def min_max_scale(series: pd.Series) -> pd.Series:
    # NOTE: Returns all-zeros for a single-element or constant series (span==0).
    # This is intentional: a single-molecule batch has no meaningful SAS range,
    # so the SAS component contributes 0.5 (i.e. neutral) after inversion in
    # compute_candidate_score. Callers that need a meaningful SAS component
    # should always pass a batch of at least 2 molecules.
    span = float(series.max() - series.min())
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def compute_candidate_score(scored: pd.DataFrame) -> pd.Series:
    risk_component = 1.0 - scored["overall_toxicity_risk"].clip(0.0, 1.0)
    qed_component = scored["qed"].clip(0.0, 1.0)
    sas_component = 1.0 - min_max_scale(scored["SAS"])
    logp_component = np.exp(-((scored["logP"] - 2.5) ** 2) / 8.0)
    return 0.50 * risk_component + 0.25 * qed_component + 0.15 * sas_component + 0.10 * logp_component
