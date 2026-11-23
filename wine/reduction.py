"""How many dimensions does wine chemistry really have, and does quality live in them?"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def components_for(X, threshold=0.9):
    """Smallest number of principal components explaining `threshold` of variance."""
    ratio = PCA().fit(X).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ratio), threshold) + 1), ratio


def top_loadings(X, columns, component=0, n=3):
    """Features with the largest absolute weight on a component."""
    w = PCA().fit(X).components_[component]
    idx = np.argsort(-np.abs(w))[:n]
    return [(columns[i], float(w[i])) for i in idx]


def separability(X, y, n_components, seed=42):
    """5-fold CV accuracy of logistic regression on the first n PCs.

    Compared with the majority-class rate, this says whether a low-dimensional
    view of the chemistry carries the label.
    """
    Z = PCA(n_components=n_components).fit_transform(X)
    clf = LogisticRegression(max_iter=2000)
    return float(cross_val_score(clf, Z, y, cv=5).mean())


def majority_rate(y):
    return float(np.bincount(np.asarray(y) - np.min(y)).max() / len(y))
