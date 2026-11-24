# Wine Cultivar Clustering

![CI](https://github.com/parsafaramarzi/wine-cultivar-clustering/actions/workflows/ci.yml/badge.svg)

**Problem:** 178 Italian wines are described by 13 chemical measurements. They come from three grape cultivars, but the cultivar label is hidden. **Can unsupervised clustering rediscover the cultivars from chemistry alone?**

**Answer:** yes with k-means (adjusted Rand index 0.90), no with DBSCAN (best ARI 0.42, and only when tuned with the answer in hand).

All features are z-scored first (proline is about 1000 times larger than hue). The cultivar label is never used for fitting, only to score the result with the adjusted Rand index (ARI: 1 is a perfect match, 0 is chance).

## Methods compared

| Method | Clusters | Noise points | Silhouette | ARI vs cultivar |
|---|---|---|---|---|
| k-means, k = 3 | 3 | 0 | 0.285 | **0.897** |
| DBSCAN, eps from k-distance elbow, min_samples = 26 | 1 | 8 | n/a | -0.007 |
| DBSCAN, best of a 30-setting grid (picked using the labels) | 2 | 32 | 0.323 | 0.424 |

PCA is used only to draw the data in 2D (PC1 vs PC2), coloured by the true cultivar, the k-means result and the best DBSCAN result:

![PCA view](results/cultivars_pca_view.png)

## What this shows

- The cultivars are nearly spherical blobs in standardised space, which is what k-means assumes. It recovers them with 0.90 ARI, though it is told k = 3.
- DBSCAN struggles. The cultivars have different densities and touch each other, so a single eps either merges them (the textbook elbow rule finds one big cluster) or marks most of the data as noise. Even the best setting in the grid, chosen by peeking at the labels, reaches only ARI 0.42 and rejects 32 wines as noise.
- The highest silhouette (0.323, DBSCAN) goes with the *worse* match to the truth. Silhouette is not a safe tuning target here.

![DBSCAN sensitivity](results/dbscan_sensitivity.png)

Full grid: [`results/dbscan_grid.csv`](results/dbscan_grid.csv). Only 6 of the 30 settings exceed ARI 0.2, and 6 label every wine as noise: DBSCAN is very sensitive to its parameters on this data.

## Limitations

- 178 wines from one region; the result says how separable these three cultivars are, not that chemistry identifies cultivars in general.
- k-means is given the true number of clusters (3). Choosing k without labels is a separate problem.
- DBSCAN was tuned over a limited grid (eps 1.5-4.0, min_samples 3-20); a finer search could do somewhat better, but the sensitivity is the finding.
- Hierarchical clustering and Gaussian mixtures are natural additions and were not tried.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_analysis.py      # regenerates everything in results/
pytest                      # 8 tests
```

## Layout

```
wine/           data loading, clustering helpers
run_analysis.py reproduces all tables and figures
tests/          unit tests
data/           wine_cultivars.csv (UCI Wine, with the cultivar label kept for scoring)
results/        figures and CSVs used in this README
```

## History

This repo replaces the earlier method-centred repo `Clustering_DBSCAN`. The wine-quality half of the old `DimReduction_PCA` repo is a different problem and lives in `wine-quality-pca`.
