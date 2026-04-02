from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import FP_BITS, KMEANS_CLUSTERS, NN_NEIGHBORS, RANDOM_STATE


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles).strip())


def morgan_fp(mol, radius: int = 2, n_bits: int = FP_BITS) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calc_descriptor_dict(mol) -> dict[str, float]:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "AromaticRings": Lipinski.NumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NHOHCount": Lipinski.NHOHCount(mol),
        "NOCount": Lipinski.NOCount(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),
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
) -> pd.DataFrame:
    descriptor_columns = list(query_desc.columns)
    query_scaled = scaler.transform(query_desc[descriptor_columns])
    distances, neighbors = nn.kneighbors(query_scaled)

    neighbor_logp = support_lookup.iloc[neighbors.flatten()]["logP"].to_numpy().reshape(len(neighbors), NN_NEIGHBORS)
    neighbor_qed = support_lookup.iloc[neighbors.flatten()]["qed"].to_numpy().reshape(len(neighbors), NN_NEIGHBORS)
    neighbor_sas = support_lookup.iloc[neighbors.flatten()]["SAS"].to_numpy().reshape(len(neighbors), NN_NEIGHBORS)
    cluster_distance = kmeans.transform(query_scaled).min(axis=1)

    return pd.DataFrame(
        {
            "zinc_nn_dist_min": distances.min(axis=1),
            "zinc_nn_dist_mean": distances.mean(axis=1),
            "zinc_nn_logP_mean": neighbor_logp.mean(axis=1),
            "zinc_nn_qed_mean": neighbor_qed.mean(axis=1),
            "zinc_nn_SAS_mean": neighbor_sas.mean(axis=1),
            "cluster_distance": cluster_distance,
        },
        index=query_desc.index,
    )


def summarize_support_distribution(tox21_support: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for column in ["zinc_nn_dist_mean", "cluster_distance"]:
        metrics[f"{column}_p50"] = float(tox21_support[column].quantile(0.50))
        metrics[f"{column}_p85"] = float(tox21_support[column].quantile(0.85))
    return metrics


def classify_risk(overall_risk: float) -> str:
    if overall_risk < 0.25:
        return "Low"
    if overall_risk < 0.50:
        return "Moderate"
    if overall_risk < 0.75:
        return "Elevated"
    return "High"


def classify_confidence(support_features: dict[str, float], reference_stats: dict[str, float]) -> str:
    dist_mean = float(support_features["zinc_nn_dist_mean"])
    cluster_distance = float(support_features["cluster_distance"])
    if (
        dist_mean <= reference_stats["zinc_nn_dist_mean_p50"]
        and cluster_distance <= reference_stats["cluster_distance_p50"]
    ):
        return "High"
    if (
        dist_mean <= reference_stats["zinc_nn_dist_mean_p85"]
        and cluster_distance <= reference_stats["cluster_distance_p85"]
    ):
        return "Moderate"
    return "Low"


def attach_feature_values(driver_table: pd.DataFrame, feature_row: pd.Series) -> pd.DataFrame:
    table = driver_table.copy()
    table["value"] = table["feature"].map(feature_row.to_dict())
    return table


def min_max_scale(series: pd.Series) -> pd.Series:
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
