"""Reproduces every table and figure in the README into results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from wine.clustering import dbscan_grid, k_distance_eps, kmeans_labels, score_labels
from wine.data import load_cultivars, scale

OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEED = 42


def part1_cultivars():
    X, truth = load_cultivars()
    Z = scale(X)
    truth = truth.to_numpy()

    # k-means with the true number of groups, as a reference
    km = kmeans_labels(Z, 3, SEED)
    rows = [{"method": "k-means (k=3)", "eps": "", "min_samples": "", **score_labels(Z, km, truth)}]

    # DBSCAN: eps from the k-distance elbow, and a grid to show sensitivity
    eps_rule = k_distance_eps(Z, 2 * Z.shape[1])  # min_samples = 2*dim rule of thumb
    ms_rule = 2 * Z.shape[1]
    db = DBSCAN(eps=eps_rule, min_samples=ms_rule).fit_predict(Z)
    rows.append({"method": "DBSCAN (elbow eps, min_samples=2d)", "eps": round(eps_rule, 2),
                 "min_samples": ms_rule, **score_labels(Z, db, truth)})

    grid = pd.DataFrame(dbscan_grid(Z, truth, np.arange(1.5, 4.01, 0.5), [3, 5, 10, 15, 20]))
    grid.to_csv(OUT / "dbscan_grid.csv", index=False)
    best = grid.sort_values("ari", ascending=False).iloc[0]
    rows.append({"method": "DBSCAN (best ARI in grid, uses labels)", "eps": best.eps,
                 "min_samples": int(best.min_samples), "clusters": int(best.clusters),
                 "noise": int(best.noise), "silhouette": best.silhouette, "ari": best.ari})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "cultivar_methods.csv", index=False)

    # figure: PCA view coloured by truth, k-means, and best DBSCAN
    P = PCA(n_components=2).fit_transform(Z)
    best_labels = DBSCAN(
        eps=best.eps, min_samples=int(best.min_samples)).fit_predict(Z)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for a, lab, title in zip(ax, [truth, km, best_labels],
                             ["True cultivar", "k-means (k=3)", "DBSCAN (best in grid)"]):
        a.scatter(P[:, 0], P[:, 1], c=lab, cmap="viridis", s=18)
        a.set_title(title)
        a.set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(OUT / "cultivars_pca_view.png", dpi=130)
    plt.close(fig)

    # heatmap of ARI over the grid
    piv = grid.pivot(index="min_samples", columns="eps", values="ari")
    fig, a = plt.subplots(figsize=(6, 4))
    im = a.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    a.set_xticks(range(len(piv.columns)), piv.columns)
    a.set_yticks(range(len(piv.index)), piv.index)
    a.set_xlabel("eps")
    a.set_ylabel("min_samples")
    a.set_title("DBSCAN: agreement with cultivars (ARI)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "dbscan_sensitivity.png", dpi=130)
    plt.close(fig)
    return table, grid


if __name__ == "__main__":
    table, grid = part1_cultivars()
    print(table.round(3).to_string(index=False))
