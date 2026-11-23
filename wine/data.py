"""Loading and scaling the two wine datasets."""
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data"


def load_cultivars(path=None):
    """UCI wine: 178 Italian wines, 13 chemical features, 3 grape cultivars.

    Returns (features, cultivar). The cultivar is never used for fitting,
    only to check the clusters afterwards.
    """
    df = pd.read_csv(path or DATA / "wine_cultivars.csv")
    return df.drop(columns="cultivar"), df["cultivar"]


def load_quality(path=None):
    """Red vinho verde: 1143 wines, 11 chemical features, sensory quality 3-8."""
    df = pd.read_csv(path or DATA / "wine_quality_red.csv")
    return df.drop(columns=["quality", "Id"]), df["quality"]


def scale(X):
    """Z-score every feature. Required: proline is ~1000x larger than hue."""
    return StandardScaler().fit_transform(X)
