# Major-Project
# Efficient Handling of Large Geospatial Datasets Using Indexing Techniques

> **IEEE-format research paper + full reproducible benchmark suite**

A comprehensive survey and performance analysis of spatial indexing techniques,
including a real-data benchmark (**GeoScale-Bench**) and an adaptive hybrid
index framework (**GHIX**) with an MLP query router.

---

## Authors

Nishant Raj, Akshita, Harmanjeet Singh, Jobandeep Singh, Akanksha, Lakshay Goel,
Pulastaya Bansal, Aditi Parmar, Parasdeep Singh

Department of AIT-CSE, Chandigarh University, Punjab, India

---

## Repository Contents

```
├── geospatial_indexing_REAL.pdf      # Final compiled IEEE paper (5 pages)
├── geospatial_indexing_REAL.tex      # LaTeX source
├── GeoIndex_Full_Benchmark.ipynb     # Complete experiment notebook (GeoScale-Bench + GHIX)
├── results.json                      # Raw experimental results (all measured numbers)
└── README.md                         # This file
```

---

## What Was Benchmarked

### Index Structures
| Index | Implementation | Notes |
|---|---|---|
| R-tree STR | libspatialindex 2.0 | STR bulk load, fill factor 0.9 |
| Hilbert R-tree | libspatialindex 2.0 | Hilbert-ordered STR, fill factor 1.0 |
| KD-tree | scipy `cKDTree` | C implementation, leafsize 16 |
| Geohash B-tree | pygeohash + bisect | Precision 7 (~150m), sorted list |
| Hilbert B-tree | NumPy + bisect | 14th-order Hilbert sort |
| PR-Quadtree | Pure Python | Capacity 16, max depth 22 |

### Datasets
| Dataset | Records | Source |
|---|---|---|
| OSM (5 cities) | 1,313,557 | OpenStreetMap via OSMnx — Chandigarh, New Delhi, Bangalore, London, New York City |
| AIS-Corridors | 999,990 | Synthetic shipping lane trajectories (15 corridors) |
| Elevation-Grid | 1,000,000 | Synthetic SRTM-style near-uniform grid |
| Synth-Uniform | 1,000,000 | IID uniform in [0,1]² |
| Synth-Gaussian | 1,000,000 | 100 clusters, σ=0.008 |
| Synth-Zipfian | 1,000,000 | Zipf α=1.5, 20 hotspots |

### Experiments
- **Range queries**: 6 selectivity levels (0.001% → 10%), 300 queries each
- **kNN queries**: k ∈ {1, 5, 10, 20, 50, 100}, 300 queries each
- **Insert throughput**: 2,000-point batches
- **Scalability**: n = 100K, 250K, 500K, 750K, 1M, 2M, 5M records
- **Distribution sensitivity**: Uniform vs. Gaussian vs. Zipfian at σ=1%

---

## Key Results

### Construction Time and Memory (OSM, n=1M)
| Index | Build (s) | Memory (MB) | RQ@0.1% (ms) | RQ@1% (ms) | kNN k=10 (ms) |
|---|---|---|---|---|---|
| R-tree STR | 2.54 | 208.6 | 0.0082 | 0.0083 | 0.0391 |
| Hilbert R-tree | 9.15 | 92.8 | 0.0052 | 0.0065 | 0.0406 |
| KD-tree | **0.24** | **10.7** | 0.0039 | 0.0038 | **0.0303** |
| Geohash B-tree | 1.64 | — | **0.0015** | **0.0015** | N/A |
| Hilbert B-tree | 6.61 | 6.3 | 0.0275 | 0.0282 | N/A |
| PR-Quadtree† | 5.05 | 15.9 | 0.0011 | 0.0012 | N/A |

†Built on 200K subset only

### Insert Throughput
| Index | TPS |
|---|---|
| R-tree STR | 47,857 |
| Hilbert R-tree | 47,181 |
| Geohash B-tree | 1,947 |
| KD-tree | N/A (no online insert) |

### Distribution Sensitivity — Range Query at σ=1% (ms)
| Index | Uniform | Gaussian | Zipfian |
|---|---|---|---|
| R-tree STR | 1.797 | 1.735 | **21.686** |
| Hilbert R-tree | 1.809 | 1.743 | **21.948** |
| KD-tree | 0.322 | 0.356 | **0.023** |
| Geohash | 0.0026 | 0.0019 | 0.0016 |
| Hilbert B-tree | 0.299 | 0.271 | 0.111 |

> R-tree variants degrade **12×** on Zipfian data. KD-tree and Geohash are robust.

### GHIX Routing Classifier
| Metric | Value |
|---|---|
| Architecture | 3-layer MLP (64–32–16, ReLU, Adam) |
| Training queries | 2,160 |
| Validation accuracy | **97.59%** |
| Inference latency | 0.071 ms |
| Training time | 0.087 s |

---

## GHIX — Geospatial Hybrid Index

GHIX maintains three concurrent indexes (R-tree STR + KD-tree + Geohash)
and uses a trained MLP classifier to route each query to the predicted
fastest index. The classifier takes 6 features per query:
query type, log-selectivity, box area, centroid (cx, cy), and k/100.

Ground-truth label distribution on OSM data (from 3,000 training queries):
- Geohash wins: 1,796 queries (59.9%)
- KD-tree wins: 615 queries (20.5%)
- R-tree STR wins: 289 queries (9.6%)

The classifier achieves 94.1% accuracy at just 50 training queries and
stabilizes at 97.59% from 1,000 queries onward.

**Note on mixed workload results:** In the 80% range / 10% kNN / 10% insert
workload on OSM data, Geohash dominates due to the prevalence of
low-selectivity range queries. GHIX provides the greatest benefit in
workloads with a more equal mix of query types.

---

## Hardware

All experiments ran on:
- **CPU**: Apple M-series ARM64, 8 cores
- **RAM**: 17.2 GB
- **OS**: macOS 26.3
- **Python**: 3.14.3
- **No GPU** — all measurements are single-threaded CPU

---

## Reproducing the Experiments

### Option 1: Google Colab (recommended)
1. Open `GeoIndex_Full_Benchmark.ipynb` in [Google Colab](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU** → Save
3. Runtime → **Run all**
4. Download `results.json` when complete (~45–60 min)

### Option 2: Local Machine
```bash
# Install dependencies
pip install rtree scipy scikit-learn osmnx pygeohash numpy pandas tqdm psutil matplotlib geopandas shapely

# On Linux, also:
apt-get install libspatialindex-dev

# Run the notebook
jupyter notebook GeoIndex_Full_Benchmark.ipynb
```

### Requirements
```
rtree
scipy
scikit-learn
osmnx
pygeohash
numpy
pandas
tqdm
psutil
matplotlib
geopandas
shapely
requests
```

---

## Compiling the Paper

Requires a full TeX Live installation with `pgfplots`, `tikz`, and `IEEEtran`.

```bash
pdflatex geospatial_indexing_REAL.tex
pdflatex geospatial_indexing_REAL.tex   # run twice for cross-references
```

The compiled PDF is also provided as `geospatial_indexing_REAL.pdf`.

---

## Selection Guidelines

| Workload | Recommended | Avoid |
|---|---|---|
| Low-selectivity range (σ < 1%) | Geohash B-tree | Hilbert B-tree |
| kNN dominated | KD-tree | Geohash (no native kNN) |
| High-selectivity range (σ > 5%) | KD-tree | R-tree variants |
| High write rate | R-tree STR | KD-tree (no insert) |
| Skewed / Zipfian data | KD-tree or Geohash | R-tree variants |
| Memory-constrained | Hilbert B-tree | R-tree STR (208 MB) |
| Mixed heterogeneous workload | GHIX | Single index |

---

## Citation

If you use this benchmark or codebase, please cite:

```bibtex
@inproceedings{raj2026geoscale,
  title     = {Efficient Handling of Large Geospatial Datasets Using Indexing
               Techniques: A Comprehensive Survey and Performance Analysis},
  author    = {Raj, Nishant and Akshita and Singh, Harmanjeet and Singh, Jobandeep
               and Akanksha and Goel, Lakshay and Bansal, Pulastaya and
               Parmar, Aditi and Singh, Parasdeep},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2026},
  institution = {Chandigarh University, Punjab, India}
}
```

---

## License

This project is released for academic use. The benchmark code is MIT licensed.
OSM data is © OpenStreetMap contributors under the [ODbL license](https://www.openstreetmap.org/copyright).
