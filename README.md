# GSoC 2026 — ML4SCI Genie Project
## Non-local GNNs for Jet Classification

**Applicant:** Ritesh Bhalerao  
**Project:** Non-local GNNs for Jet Classification  
**Dataset:** Quark/Gluon jet events — 3-channel images (ECAL, HCAL, Tracks), 125×125 pixels

---

## Tasks Completed

### Common Task 1 — Autoencoder
Train an autoencoder to learn latent representations of 3-channel jet images.

- Architecture: Conv2d encoder (3→32→64→128) + bottleneck + ConvTranspose2d decoder
- Preprocessing: log1p scaling, 99.9th percentile clipping, per-image normalization
- Custom loss: signal-focused (5× weight on non-zero pixels) + background penalty + sparsity
- **Mean MSE: 0.004065** | ECAL: 0.008558 | HCAL: 0.002655 | Tracks: 0.000982

### Common Task 2 — Baseline GNN (Jets as Graphs)
Classify quark/gluon jets using a graph neural network.

- Point cloud: non-zero pixels only, 6 node features (x, y, energy, log-energy, frac-energy, channel)
- Graph: 8-nearest-neighbours in pixel space (manual KNN, no torch-cluster dependency)
- Model: 3-layer GraphSAGE (6→64→128→128) + global mean pooling + MLP head
- **ROC-AUC: 0.7443**

### Specific Task 4 — Non-local GNN
Build a non-local GNN and compare with baseline using ROC-AUC.

- Architecture: Same SAGEConv backbone + per-graph scaled dot-product self-attention block
- Non-local block: attention computed within each jet independently (correct, no cross-jet leakage)
- Scale factor: √128 ≈ 11.31 (standard scaled dot-product attention)
- **Baseline GNN ROC-AUC: 0.7443 | Non-local GNN ROC-AUC: see notebook**

---

## Repository Structure

```
├── Genie_gnn_jet_classification.ipynb   # Main notebook — all 3 tasks
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── results/                             # Saved plots and metrics
│   ├── ae_loss_curve.png
│   ├── ae_reconstruction.png
│   ├── ae_per_channel.png
│   ├── ae_mse_distribution.png
│   └── task4_comparison.png
└── data/
    └── .gitkeep                         # Data not tracked (too large)
```

---

## How to Run

### On Google Colab (recommended)
1. Open `Genie_gnn_jet_classification.ipynb` in Google Colab
2. Mount your Google Drive
3. The notebook downloads the dataset automatically via `gdown`
4. Run all cells top to bottom

### Dataset
The dataset is downloaded automatically in the notebook:
```python
file_id = "1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "data.hdf5")
```
Do **not** commit `data.hdf5` to this repo (701 MB).

---

## Dependencies
See `requirements.txt`. Key packages:
- `torch`, `torch-geometric`
- `h5py`, `numpy`, `scikit-learn`
- `matplotlib`, `gdown`

---

## Results Summary

| Task | Metric | Value |
|------|--------|-------|
| Task 1 — Autoencoder | Mean MSE | 0.004065 |
| Task 1 — Autoencoder | RMSE | 0.063757 |
| Task 2 — Baseline GNN | ROC-AUC | 0.7443 |
| Task 4 — Non-local GNN | ROC-AUC | (see notebook) |

---

## Key Design Decisions

**Why GraphSAGE?**  
SAGEConv aggregates neighbour features inductively — no fixed adjacency matrix — making it well-suited for jets of variable node count.

**Why non-local attention?**  
Local k-NN message passing (k=8) can only propagate information 3 hops in 3 layers. For a 500-node jet graph that leaves many long-range correlations uncaptured. The non-local block gives every node direct access to every other node in its jet via scaled dot-product self-attention.

**Why per-graph attention (not batch-level)?**  
Each jet is physically independent. Computing attention across all jets in a batch would allow nodes from one jet to influence nodes from another — physically meaningless and empirically harmful to performance.
