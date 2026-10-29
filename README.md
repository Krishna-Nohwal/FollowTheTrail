<div align="center">

# FollowTheTrail: Representation Trajectories for Generalizable Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-%3E%3D4.45%20%3C5-yellow)](https://huggingface.co/docs/transformers)
![Status](https://img.shields.io/badge/Status-Research%20Code-lightgrey)

**Accepted at WACV 2027**

</div>

---

## 📑 Table of Contents

- [🧭 Introduction](#-introduction)
- [✨ Highlights](#-highlights)
- [⚙️ Version and Model Details](#️-version-and-model-details)
- [🚀 Quick Start](#-quick-start)
  - [1. Installation](#1-installation)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Training](#3-training)
  - [4. Evaluation](#4-evaluation)
  - [5. Optional Frame Extraction](#5-optional-frame-extraction)
- [🗂️ Repository Layout](#️-repository-layout)
- [📚 Citation](#-citation)

## 🧭 Introduction

FollowTheTrail studies how real and AI-generated face representations change across the layers of a vision transformer. It treats selected intermediate `[CLS]` representations as points on a hypersphere, forms a trajectory feature from their spherical geometry, and uses a direction-free turning-cosine margin objective alongside classification losses.

This repository contains the DINOv3-Large training entry point, its matching multi-dataset evaluator, and optional video frame-extraction utilities. Dataset contents, pretrained weights, and trained checkpoints are not bundled.

## ✨ Highlights

- 🧭 **Representation trajectories:** spherical log maps and parallel transport describe layer-to-layer changes.
- 📐 **Turning-cosine supervision:** encourages a margin between the class-mean turning responses without fixing which class must turn in a particular direction.
- 🧠 **Parameter-efficient tuning:** trains LoRA adapters and task heads with a DINOv3-Large backbone.
- 🌍 **Cross-dataset evaluation:** includes a checkpoint evaluator for common face-forgery benchmarks.

## ⚙️ Version and Model Details

We trained on a NVIDIA A100.

| Component | Version or default |
|---|---:|
| Python | 3.10 or newer |
| PyTorch | 2.1 or newer; install the build matching your CUDA setup |
| Transformers | `>=4.45,<5` |
| PEFT | `>=0.13,<0.20` |
| timm | `>=1.0` |
| Backbone | DINOv3-Large, 24 transformer blocks |
| Trajectory blocks | `0, 8, 23` |
| Training epochs | `5` (paper protocol and repository default) |
| Image size | `256` pixels |
| Batch size | `32` |
| LoRA rank / alpha | `32 / 64` |

## 🚀 Quick Start

### 1. Installation

Use Python 3.10 or newer. First install the PyTorch build appropriate for your operating system and CUDA version using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then install the project dependencies:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the remaining requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the pretrained DINOv3 backbone through `timm`. Please follow the backbone publisher’s access and model-weight terms.

### 2. Data Preparation

Dataset files belong under `data/`, which is excluded from Git. The trainer expects a CSV manifest with `sample_dir` and `label` columns. Each sample directory must contain an `image.png`; relative `sample_dir` paths are resolved under the selected dataset root.

```text
data/
├── ffpp/
│   ├── manifest.csv
│   └── <sample_dir>/image.png
└── cdfv1/
    ├── manifest.csv
    └── <sample_dir>/image.png
```

- **FF++ training labels:** `0` = real, `1` = fake.
- **CDFv1 input labels:** the source manifest convention is `1` = real, `0` = fake; the training/evaluation code converts these labels internally.
- Preserve the official train/test separation when preparing manifests. Obtain datasets from their official sources and follow their licenses and protocols.

See [`data/README.md`](data/README.md) for the short data note.

### 3. Training

The paper’s main protocol uses five training epochs. Run from the repository root and provide paths to your local manifests and frames:

```bash
python traj.py \
  --epochs 5 \
  --manifest data/ffpp/manifest.csv \
  --root_dir data/ffpp \
  --cdfv1_csv data/cdfv1/manifest.csv \
  --cdfv1_root data/cdfv1 \
  --save_root checkpoints/followthetrail
```

The trainer saves the best and latest checkpoints, LoRA weights, and a training-history CSV under `checkpoints/followthetrail/`. These generated files are ignored by Git.

Explore all training options with:

```bash
python traj.py --help
```

### 4. Evaluation

The evaluator reads the trajectory configuration stored in the checkpoint. It supports FF++ validation, CDFv1, CDFv2, CDF++, DFD, DFDC, DFo, WDF, and UADFV. Point the options at datasets that are available locally; unavailable datasets are skipped.

```bash
python evaluate_traj_remote_all.py \
  --checkpoint checkpoints/followthetrail/best.pth \
  --ffpp_manifest data/ffpp/manifest.csv \
  --ffpp_root data/ffpp \
  --cdfv1_csv data/cdfv1/manifest.csv \
  --cdfv1_root data/cdfv1 \
  --cdfv2_fake_root data/cdfv2/fake \
  --cdfv2_real_root data/cdfv2/real \
  --dfo_fake_root data/dfo/fake \
  --dfo_real_root data/dfo/real \
  --uadfv_fake_root data/uadfv/fake \
  --uadfv_real_root data/uadfv/real \
  --output_csv results/cross_dataset.csv
```

See the complete set of dataset path options with:

```bash
python evaluate_traj_remote_all.py --help
```

### 5. Optional Frame Extraction

For the video datasets used by the optional preparation helpers:

```bash
python extract_genvideo_frames.py --input_root data/GenVideo --output_root data/genvideo
python extract_dvf_tiny_frames.py --input_root data/DVF_tiny --output_root data/dvf
```

These scripts extract frames from data you already have; they do not download datasets.

## 🗂️ Repository Layout

```text
FollowTheTrail/
├── traj.py                       # training
├── augmentations.py              # shared image loading and augmentation
├── evaluate_traj_remote_all.py   # multi-dataset checkpoint evaluation
├── extract_genvideo_frames.py   # optional GenVideo frame extraction
├── extract_dvf_tiny_frames.py   # optional DVF frame extraction
├── requirements.txt
└── data/
    └── README.md                 # expected manifest/layout notes
```


## 📚 Citation

If you use this work, please cite the paper:

```bibtex
TODO
```

No license has been selected for this code yet. The authors should add their chosen license before public release.
#
