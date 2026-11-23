import numpy as np
import pytest

from wine.clustering import dbscan_grid, k_distance_eps, kmeans_labels, score_labels
from wine.data import load_cultivars, load_quality, scale
from wine.reduction import components_for, majority_rate, separability, top_loadings


def test_dataset_shapes():
    X, c = load_cultivars()
    assert X.shape == (178, 13) and set(c) == {0, 1, 2}
    Xq, q = load_quality()
    assert Xq.shape[1] == 11 and "quality" not in Xq.columns and "Id" not in Xq.columns
    assert len(Xq) == len(q)


def test_scale_zero_mean_unit_std():
    Z = scale(load_cultivars()[0])
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-9)


def test_kmeans_recovers_cultivars():
    X, truth = load_cultivars()
    labels = kmeans_labels(scale(X), 3, 42)
    assert score_labels(scale(X), labels, truth.to_numpy())["ari"] > 0.8


def test_kmeans_deterministic():
    Z = scale(load_cultivars()[0])
    assert np.array_equal(kmeans_labels(Z, 3, 1), kmeans_labels(Z, 3, 1))


def test_score_labels_counts_noise():
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11], [50, 50]], dtype=float)
    labels = np.array([0, 0, 1, 1, -1])
    r = score_labels(X, labels, np.array([0, 0, 1, 1, 1]))
    assert r["clusters"] == 2 and r["noise"] == 1 and r["silhouette"] > 0.9


def test_score_labels_single_cluster_has_nan_silhouette():
    X = np.random.default_rng(0).normal(size=(20, 2))
    r = score_labels(X, np.zeros(20, dtype=int), np.zeros(20, dtype=int))
    assert np.isnan(r["silhouette"])


def test_k_distance_eps_positive_and_bounded():
    Z = scale(load_cultivars()[0])
    eps = k_distance_eps(Z, 5)
    assert 0 < eps < 20


def test_dbscan_grid_rows():
    Z = scale(load_cultivars()[0])
    truth = load_cultivars()[1].to_numpy()
    rows = dbscan_grid(Z, truth, [2.0, 3.0], [5, 10])
    assert len(rows) == 4 and {"eps", "min_samples", "clusters", "noise", "ari"} <= rows[0].keys()


def test_components_for_monotone_threshold():
    Z = scale(load_quality()[0])
    n90, ratio = components_for(Z, 0.9)
    n50, _ = components_for(Z, 0.5)
    assert n50 < n90 <= Z.shape[1]
    assert ratio.sum() == pytest.approx(1.0)


def test_top_loadings_sorted_by_magnitude():
    X, _ = load_quality()
    load = top_loadings(scale(X), list(X.columns), 0, 3)
    mags = [abs(w) for _, w in load]
    assert mags == sorted(mags, reverse=True) and len(load) == 3


def test_separability_in_unit_interval_and_majority():
    X, y = load_quality()
    acc = separability(scale(X), y, 2)
    assert 0 < acc <= 1
    assert 0.4 < majority_rate(y) < 0.45
