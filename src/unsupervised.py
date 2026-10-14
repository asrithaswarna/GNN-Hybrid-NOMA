"""Reproducible exploratory clustering and PCA for wireless user features."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import FEATURE_COLUMNS


def run_clustering(frame: pd.DataFrame, output_dir: str, seed: int = 42) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values = StandardScaler().fit_transform(frame[FEATURE_COLUMNS])
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(values)
    result = frame[["scenario_id", "user_id"]].copy()
    result["kmeans_cluster"] = KMeans(n_clusters=min(3, max(2, len(result) // 10)), random_state=seed, n_init=10).fit_predict(values)
    result["dbscan_cluster"] = DBSCAN(eps=1.2, min_samples=3).fit_predict(values)
    result["pca_1"] = projected[:, 0]
    result["pca_2"] = projected[:, 1]
    result.to_csv(output / "cluster_assignments.csv", index=False)
    pd.DataFrame({"component": [1, 2], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(output / "pca_explained_variance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(result["pca_1"], result["pca_2"], c=result["kmeans_cluster"], cmap="viridis")
    ax.set_title("Wireless User Clusters in PCA Space")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    fig.tight_layout()
    fig.savefig(output / "clusters_pca.png", dpi=150)
    plt.close(fig)
    return {"assignments": result, "explained_variance": pca.explained_variance_ratio_.tolist()}
