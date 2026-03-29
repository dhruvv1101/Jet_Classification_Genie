# Jet Classification — GENIE Project (GSoC 2026)

This repository contains the source code and results for the **GENIE Jet Classification** evaluation tasks, developed as part of the project **[Graph Neural Networks for Jet Classification — GENIE / ML4SCI GSoC 2026](https://ml4sci.org/gsoc/2026/proposal_GENIE1.html)**.

---

## Directory Structure

```
.
├── assets/
│   ├── ae_loss_curve.png
│   ├── ANN_Reconstructed_Images.png
│   ├── Per_Image_reconstructed_images.png
│   └── Comparision_NonLocalGnn_vs_Gnn.png
├── Genie_gnn_jet_classification.ipynb
└── README.md
```

---

## Overview

Jets are collimated sprays of particles produced in high-energy collisions at the LHC. Distinguishing quark-initiated jets from gluon-initiated jets is a fundamental task in high energy physics. This project applies deep learning — specifically **Convolutional Autoencoders** and **Graph Neural Networks** — to jet images from the GENIE dataset.

| Task | Description | Method |
|------|-------------|--------|
| Common Task 1 | Jet Image Compression & Reconstruction | Convolutional Autoencoder |
| Common Task 2 | Quark/Gluon Jet Classification (Baseline) | GraphSAGE GNN |
| Specific Task 4 | Non-local GNN for Jet Classification | GraphSAGE + Self-Attention |

---

## Common Task 1 — Autoencoder for Jet Image Reconstruction

### Approach

Jet images are 3-channel 125×125 arrays (ECAL, HCAL, Tracks). The autoencoder learns a compressed latent representation and reconstructs the original jet energy deposits.

**Preprocessing:**
- Log-transform: `X = log1p(X)` to compress dynamic range
- Clip outliers at the 99.9th percentile
- Per-image normalization to `[0, 1]`

**Architecture:**

| Block | Layers | Channels | Activation |
|-------|--------|----------|------------|
| Encoder | Conv2d ×3 (stride=2) | 3 → 32 → 64 → 128 | LeakyReLU + BatchNorm |
| Bottleneck | Conv2d ×1 | 128 → 128 | LeakyReLU |
| Decoder | ConvTranspose2d ×3 (stride=2) | 128 → 64 → 32 → 3 | LeakyReLU + BatchNorm |

**Custom Loss Function:**

A physics-motivated weighted MSE was used to prioritize accurate reconstruction of signal deposits over empty background:

```
Loss = 5 × MSE(signal pixels) + 0.1 × MSE(background pixels) + 0.001 × L1 sparsity
```

**Training:**
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Epochs: 30
- Scheduler: ReduceLROnPlateau

### Training Curve

![Autoencoder Training Curve](assets/ae_loss_curve.png)

Both train loss and validation MSE converge smoothly with no overfitting. The gap between train and val curves narrows progressively over 30 epochs, settling near ~0.0002 (train) and ~0.0008 (val MSE).

### Reconstructed Images

![Reconstructed Jet Images](assets/ANN_Reconstructed_Images.png)

The autoencoder successfully localises the jet core and key energy deposits. The reconstructions preserve the spatial structure of the jet, though fine-grained noise in the tails is smoothed — a consequence of the sparsity regularizer and signal-weighted loss.

### Per-Image Reconstruction MSE Distribution

![Per-Image MSE Distribution](assets/Per_Image_reconstructed_images.png)

| Metric | Value |
|--------|-------|
| Mean per-image MSE | **0.0041** |
| Distribution | Right-skewed; bulk of images reconstruct with MSE < 0.006 |
| Outliers (MSE > 0.015) | Rare — correspond to jets with unusual energy topologies |

---

## Common Task 2 — Baseline GNN for Jet Classification

### Approach

Jet images are converted from pixel grids into sparse point-cloud graphs, allowing a GNN to exploit the local spatial structure of energy deposits.

**Graph Construction:**
- **Nodes:** Non-zero pixels only (sparse representation). Each node has **6 features**:

| Feature | Description |
|---------|-------------|
| `x` | Normalised pixel column position |
| `y` | Normalised pixel row position |
| `E` | Raw energy deposit |
| `log(E)` | Log-energy (reduces dynamic range) |
| `E_frac` | Fractional energy (E / total jet E) |
| `channel` | Detector channel id (ECAL=0, HCAL=1, Track=2) |

- **Edges:** 8-nearest-neighbours in (x, y) pixel space via Euclidean distance. k=8 gives local spatial context without introducing long-range noise.

**Model Architecture — GraphSAGE:**

```
Input (6) → SAGEConv → BN → ReLU (64)
          → SAGEConv → BN → ReLU (128)
          → SAGEConv → BN → ReLU (128)
          → GlobalMeanPool
          → Linear(128→64) → ReLU → Dropout(0.5)
          → Linear(64→2)
```

SAGEConv was chosen because it aggregates neighbor features inductively (no fixed adjacency matrix), making it well-suited for jets of variable size.

**Training:**
- Optimizer: Adam (lr=3e-4, weight_decay=1e-5)
- Loss: CrossEntropyLoss with class-balancing weights + label smoothing (0.05)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Gradient clipping: norm=1.0
- Epochs: 17
- Data: 10,000 graphs → 80/20 train/test split

### Results

| Model | ROC-AUC | Epochs | Training Set Size |
|-------|---------|--------|-------------------|
| Baseline GNN (GraphSAGE) | **0.7842** | 17 | 8,000 jets |

**Interpretation:** An AUC of 0.7842 is competitive for a CPU-constrained run on 8,000 jets. The theoretical ceiling for this dataset with state-of-the-art models is ~0.83–0.87 (ParticleNet, JEDI-net). The gap is primarily explained by:
1. Training set size — production runs use 100k+ jets
2. CPU-only training limiting epochs and model depth
3. No data augmentation or edge reweighting

The model generalises well — train and test loss are close — confirming that Dropout(0.5) and class-balanced loss are working correctly.

---

## Specific Task 4 — Non-local GNN

### Approach

Standard GNNs are limited to local message-passing along edges. A **Non-local block** (scaled dot-product self-attention) is added after the GraphSAGE layers, enabling every node in a jet to directly attend to every other node, capturing long-range dependencies that local message-passing misses.

**Architecture — Non-local GNN:**

```
Input (6) → SAGEConv → BN → ReLU (64)
          → SAGEConv → BN → ReLU (128)
          → SAGEConv → BN → ReLU (128)
          → Non-local Self-Attention Block  ← key addition
          → GlobalMeanPool
          → Linear(128→64) → ReLU → Dropout(0.4)
          → Linear(64→2)
```

**Non-local Block (per-jet self-attention):**

For each jet in the batch independently:
1. Extract node features → `(N_jet, d)`
2. Compute Q, K, V projections
3. Attention: `softmax(QKᵀ / √d) · V`
4. Residual: `x = x + attended`

This is correct non-local / self-attention — long-range within each jet, with no cross-contamination between jets in the batch, and fully differentiable.

### Results

| Model | ROC-AUC | Epochs |
|-------|---------|--------|
| Baseline GNN | 0.7842 | 17 |
| Non-local GNN | **0.7840** | 17 |

### Comparison

![Non-local GNN vs Baseline Comparison](assets/Comparision_NonLocalGnn_vs_Gnn.png)

**ROC Curves (left):** Both models achieve nearly identical AUC (~0.784), with the ROC curves overlapping throughout. This suggests the non-local block does not hurt performance.

**Training Loss Curves (right):** The Non-local GNN (orange) converges slightly faster and achieves a marginally lower final training loss (~0.597 vs ~0.600), indicating the self-attention block captures additional structure. The AUC parity is likely due to the limited training set size (8,000 jets) — at larger scale, the non-local block is expected to show greater gains.

---

## Environment Setup

```bash
# Clone the repository
git clone https://github.com/dhruvv1101/Jet_Classification_Genie.git
cd Jet_Classification_Genie
```

### Install Dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
pip install awkward uproot h5py tqdm dask[complete] pyarrow gdown scikit-learn matplotlib
```

### Run the Notebook

Open `Genie_gnn_jet_classification.ipynb` in Google Colab or Jupyter:

```bash
jupyter notebook Genie_gnn_jet_classification.ipynb
```

> **Note:** The notebook mounts Google Drive to load the GENIE jet dataset. Update `base_path` in Cell 2 to point to your own Drive folder containing the dataset files.

---

## Task Summary

| Task | Method | Key Metric | Result |
|------|--------|------------|--------|
| Common Task 1 | Convolutional Autoencoder | Mean per-image MSE | **0.0041** |
| Common Task 2 | GraphSAGE GNN (3 layers, k=8 edges) | ROC-AUC | **0.7842** |
| Specific Task 4 | GraphSAGE + Non-local Self-Attention | ROC-AUC | **0.7840** |

---

## References

- Moreno et al., *JEDI-net: a jet identification algorithm based on interaction networks* (2020)
- Qu & Gouskos, *ParticleNet: Jet Tagging via Particle Clouds* (2020)
- Wang et al., *Non-local Neural Networks* (CVPR 2018)
- Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE, NeurIPS 2017)
