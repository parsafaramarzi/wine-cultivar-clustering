"""Can unsupervised clustering rediscover the grape cultivars?"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


def k_distance_eps(X, k):
    """Heuristic eps: the point of maximum curvature of the sorted k-NN distances."""
    dist, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    d = np.sort(dist[:, -1])
    x = np.arange(len(d))
    p1, p2 = np.array([0, d[0]]), np.array([len(d) - 1, d[-1]])
    v = (p2 - p1) / np.linalg.norm(p2 - p1)
    pts = np.column_stack([x, d]) - p1
    perp = pts - np.outer(pts @ v, v)
    return float(d[np.argmax(np.linalg.norm(perp, axis=1))])


def score_labels(X, labels, truth):
    """Summary of a clustering. Noise (-1) counts as its own group for ARI."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    sil = silhouette_score(X[mask], labels[mask]) if n_clusters > 1 else float("nan")
    return {
        "clusters": n_clusters,
        "noise": int((~mask).sum()),
        "silhouette": sil,
        "ari": adjusted_rand_score(truth, labels),
    }


def dbscan_grid(X, truth, eps_values, min_samples_values):
    """Fit DBSCAN over a grid. Returns list of dicts (one per setting)."""
    rows = []
    for eps in eps_values:
        for ms in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            rows.append({"eps": round(float(eps), 2), "min_samples": ms,
                         **score_labels(X, labels, truth)})
    return rows


def kmeans_labels(X, k=3, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
