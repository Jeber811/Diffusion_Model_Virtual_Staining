# Comparative Study of Diffusion-Based Architectures for H&E-to-IHC Virtual Staining

**CAP5516 — University of Central Florida**
**Author:** Jake Weber

---

## Overview

This repository contains the full implementation of four diffusion-based generative models for **H&E-to-IHC virtual staining** — the task of computationally predicting Immunohistochemistry (IHC) stained images directly from Hematoxylin and Eosin (H&E) histology slides. Virtual staining has significant clinical potential, as it could eliminate the need for repeated physical staining procedures, reduce laboratory cost, and accelerate HER2 scoring in breast cancer diagnosis.

All four models are evaluated on the [BCI-512 benchmark dataset](https://bupt-ai-cz.github.io/BCI/) under standardized 2-hour A100 training budgets and compared on PSNR, SSIM, LPIPS, and FID.

> **Note:** This project was completed as part of a course at UCF and represents an exploratory study. All models are partially trained (2-hour wall-clock cap) and results reflect that constraint.

---

## Models

| Model | Description | Params |
|---|---|---|
| **BBDM** | Brownian Bridge Diffusion Model — defines a direct stochastic bridge between H&E and IHC image domains | 120.25M |
| **PST-Diff+STN** | UNet-based conditional diffusion with Spatial Transformer Network for slide misalignment correction and Chromatic Frequency Guidance loss | 69.43M |
| **ScoreTopo** | Score-based diffusion with a novel Topological Consistency Loss (soft Euler characteristic) and Mutual Information contrastive objective | 66.74M |
| **HistDiT+SAM** | Diffusion Transformer (DiT) with dual-stream SAM-proxy conditioning for structure-aware generation | 80.66M |

---

## Results

All models evaluated on the BCI validation/test set. ↑ = higher is better, ↓ = lower is better.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Epochs | N |
|---|---|---|---|---|---|---|
| BBDM | **15.1564** | **0.242925** | 0.7570 | 360.96 | 47 | 977 |
| PST-Diff+STN | 10.0750 | 0.015514 | **0.6643** | **214.77** | 35 | 585 |
| ScoreTopo | 6.8935 | 0.001263 | 1.1689 | 416.81 | 40 | 585 |
| HistDiT+SAM | 11.9324 | 0.000409 | 0.9403 | 443.43 | 50 | 585 |

**Key takeaway:** BBDM achieves the best pixel-level fidelity. PST-Diff+STN achieves the best perceptual distribution metrics, driven by its STN misalignment correction. ScoreTopo and HistDiT+SAM require longer training schedules to converge than the 2-hour budget allows.

---

## Repository Structure

```
Diffusion_Model_Virtual_Staining/
│
├── BBDM/                          # BBDM model (adapted from official repo)
│   ├── configs/
│   │   └── My_Method_config.yaml  # BCI-512 dataset configuration
│   ├── datasets/
│   │   └── custom_aligned.py      # Custom BCI dataset loader
│   └── main.py                    # Entry point (from official BBDM repo)
│
├── notebooks/
│   ├── Cell_7_PST_Diff_STN.ipynb  # PST-Diff+STN training notebook
│   ├── Cell_8_ScoreTopo.ipynb     # ScoreTopo training notebook
│   ├── Cell_9_HistDiT_SAM.ipynb   # HistDiT+SAM training notebook
│   └── Evaluation.ipynb           # Full evaluation cell (all 4 models)
│
├── eval/
│   └── eval_cell.py               # Standalone evaluation script
│                                  # Set USE_HARDCODED = True/False
│
├── scripts/
│   └── prepare_dataset.sh         # Dataset download and preprocessing
│
└── README.md
```

---

## Dataset Setup

This project uses the **BCI (Breast Cancer Immunohistochemical Image Generation)** dataset.

### Download

The BCI dataset is publicly available at the [official BCI Challenge page](https://bupt-ai-cz.github.io/BCI/).

After downloading, organize the dataset as follows:

```
datasets_standardized/
└── BCI_512/
    ├── train/
    │   ├── A/          # H&E images  (3,311 patches)
    │   └── B/          # IHC images  (3,311 patches)
    ├── val/
    │   ├── A/          # H&E images  (585 patches)
    │   └── B/          # IHC images  (585 patches)
    └── test/
        ├── A/          # H&E images  (977 patches)
        └── B/          # IHC images  (977 patches)
```

### File Naming Convention

H&E images are named `XXXX_A.png` and their corresponding IHC images are named `XXXX_B.png`. The dataset loaders expect this convention.

### Google Drive (for Colab)

For running on Google Colab, upload the dataset archive to Google Drive:

```
/content/drive/MyDrive/BCI_Project_Final/BCI_512_Standardized.tar.gz
```

Then extract with:

```python
import tarfile
with tarfile.open("/content/drive/MyDrive/BCI_Project_Final/BCI_512_Standardized.tar.gz") as f:
    f.extractall("/content/datasets_standardized/")
```

---

## Requirements

```
torch>=2.0.0
torchvision
numpy
Pillow
lpips
pytorch-fid
pyyaml
```

Install with:

```bash
pip install torch torchvision numpy Pillow lpips pytorch-fid pyyaml
```

For Colab, all dependencies are installed automatically at the top of each notebook.

---

## Training

All training was performed on **NVIDIA A100-SXM4-80GB** via Google Colab Pro with a 2-hour wall-clock budget per model. Each model is contained in its own notebook.

### BBDM

BBDM uses the [official BBDM repository](https://github.com/xuekt98/BBDM) with a custom BCI dataset loader.

```bash
cd BBDM
python main.py \
  --config configs/My_Method_config.yaml \
  --train \
  --gpu_ids 0
```

Checkpoints are saved to:
```
BBDM/results/BCI_512/BBDM_A100/checkpoint/
```

### PST-Diff+STN, ScoreTopo, HistDiT+SAM

Open the corresponding notebook in Google Colab:

1. Mount Google Drive
2. Set `MODE = "a100"` at the top of the notebook
3. Run all cells — the training loop includes automatic checkpoint saving to Drive and a 2-hour wall-clock timeout

**Drive checkpoint paths:**
```
/content/drive/MyDrive/PST_STN_checkpoints_a100/final_epoch35.pt
/content/drive/MyDrive/ScoreTopo_checkpoints_a100/final_epoch40.pt
/content/drive/MyDrive/HistDiT_SAM_checkpoints_a100/epoch_0050.pt
```

---

## Evaluation

The full evaluation pipeline is in `eval/eval_cell.py` and `notebooks/Evaluation.ipynb`.

### Quick Start (Hardcoded Results)

To reproduce the results table without re-running inference:

```python
USE_HARDCODED = True   # Uses saved results instantly
```

Run the evaluation cell — this prints the full results table and LaTeX output in seconds.

### Full Inference (Reproduce from Checkpoints)

To re-run full inference from saved checkpoints:

```python
USE_HARDCODED = False  # Re-runs all inference
MAX_SCORE_IMAGES = 585 # Reduce to 50 if ScoreTopo inference is too slow (~1 min/image)
```

**Prerequisites for full inference:**

| Model | Prerequisite |
|---|---|
| BBDM | Run `python main.py --sample_to_eval` first |
| PST-Diff+STN | Re-run Cell 7 notebook with `MODE='a100'` to define classes |
| ScoreTopo | Re-run Cell 8 notebook with `MODE='a100'`, `NGF=128` |
| HistDiT+SAM | Re-run Cell 9 notebook with `MODE='a100'` |

> **Warning:** ScoreTopo inference runs at approximately 1 minute per image on an A100. Full 585-image evaluation takes ~10 hours. Use `MAX_SCORE_IMAGES = 50` for a faster approximation.

### Metrics

All four metrics are computed automatically:

- **PSNR** — Peak Signal-to-Noise Ratio
- **SSIM** — Structural Similarity Index
- **LPIPS** — Learned Perceptual Image Patch Similarity (AlexNet)
- **FID** — Fréchet Inception Distance (InceptionV3, via `pytorch-fid`)

---

## Novel Contributions

The following components are original to this study (not from prior codebases):

- **Topological Consistency Loss** — soft Euler characteristic computed at K=10 binarization thresholds, used in ScoreTopo to preserve histological structure during diffusion
- **MI Contrastive Loss** — InfoNCE-based mutual information objective between H&E encoder features and predicted IHC features (ScoreTopo)
- **STN-in-the-Loop Training** — Spatial Transformer Network embedded in the diffusion training loop of PST-Diff+STN for geometry-corrected supervision
- **Dual-Stream SAM Conditioning** — Lightweight SAM proxy encoder providing independent structural (spatial cross-attention) and content (HE encoder cross-attention) conditioning streams in HistDiT+SAM
- **Asymmetric Attention Module (AAM)** — Gated cross-attention between UNet decoder states and H&E encoder features that maintains asymmetric resolution to avoid memory explosion (PST-Diff+STN)

---

## Architecture Notes

### HEEncoder (Shared)
All models use a shared multi-scale H&E encoder:
- 3 stages: `[NGF, NGF×2, NGF×4]` channels
- GroupNorm + Conv2d + SiLU blocks
- NGF = 64 for HistDiT, NGF = 128 for ScoreTopo

### Common Training Settings
```
Optimizer:      AdamW (β₁=0.9, β₂=0.999, wd=1e-4)
Learning rate:  1e-4 → 1e-6 (cosine annealing)
Grad clip:      1.0 (L∞)
Mixed precision: bfloat16 (A100)
DDIM steps:     50 (inference)
T:              1000
```

---

## Known Issues and Quirks

- **ScoreTopo class conflict**: Cell 8 (ScoreTopo) and Cell 9 (HistDiT) both define `HEEncoder` with different architectures. Run the eval cell immediately after whichever model cell you need without running the other in between, or use `he_enc = HEEncoder(64).to(DEVICE)` to explicitly rebuild for HistDiT.

- **HistDiT target resize**: The HistDiT evaluation saves reconstructed images at 256×256 (downsampled from 512px). The target IHC images must also be resized to 256×256 before running `compute_metrics`. The evaluation notebook handles this automatically.

- **ScoreTopo NGF mismatch**: The ScoreTopo checkpoint was trained with `NGF=128`. When re-running Cell 8 to define classes, set `NGF=128` in the a100 config block, not the default value.

---

## References

- [BCI Dataset & Challenge](https://bupt-ai-cz.github.io/BCI/) — Liu et al., CVPR 2022
- [BBDM](https://arxiv.org/abs/2205.07680) — Li et al., CVPR 2023
- [GANs vs. Diffusion Models for Virtual Staining](https://arxiv.org/abs/2506.18484) — 2025 *(primary comparative reference)*
- [DiT](https://arxiv.org/abs/2212.09748) — Peebles & Xie, ICCV 2023
- [HistDiT](https://arxiv.org/abs/2604.08305) — 2025
- [Segment Anything](https://arxiv.org/abs/2304.02643) — Kirillov et al., ICCV 2023
- [PST-Diff](https://ieeexplore.ieee.org/document/10816688) — 2024

---

## Acknowledgments

Training was conducted using Google Colab Pro (A100 GPUs). Early prototyping used the UCF ARCC Newton HPC cluster (H100 GPUs). The BCI dataset is provided by the BCI Challenge organizers.

---

## License

This project is for academic and research purposes. BBDM code is adapted from the [official BBDM repository](https://github.com/xuekt98/BBDM) under its original license. All original implementations (PST-Diff+STN, ScoreTopo, HistDiT+SAM) are released under MIT License.
