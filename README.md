# WPGRec: Wavelet Packet Guided Graph-Enhanced Sequential Recommendation

> **Acknowledgement.**  
> We thank the reviewers, the community, and open-source contributors for their constructive feedback and long-term support of sequential recommendation research. We release this repository to facilitate reproducibility and further exploration. If you use this codebase in your research or production, please feel free to open issues or submit pull requests.

This repository provides the **official PyTorch implementation** of **WPGRec (Wavelet Packet Guided Graph-Enhanced Sequential Recommendation)**, reproducing the core model components and the training/evaluation pipeline described in our paper. WPGRec targets **sequential recommendation**: given a user’s historical interaction sequence, it learns user preference representations and predicts the next most likely item.

---

## Paper Overview

A key challenge in sequential recommendation is that user behavior sequences contain **multi-scale preference signals** (e.g., short-term interest fluctuations, periodic patterns, and long-term stable preferences), and these signals are coupled with the global structure of the user–item interaction graph. Pure sequence encoders may fail to fully capture patterns at different temporal scales, while graph-only propagation can dilute local dynamics within sequences.

WPGRec views user sequences as **multi-resolution signals**. It applies **wavelet packet decomposition** to split sequence representations into multiple **equal-length, aligned subband components**, models both sequential and graph information **at the subband level**, and then performs adaptive fusion. This design helps preserve:
- local / short-term dynamics (high-frequency subbands),
- mid-term trends (mid-frequency subbands),
- long-term stable preference components (low-frequency subbands).

---

## Method at a Glance

WPGRec consists of three main parts:

### 1) Full-Tree SWPT Decomposition (Sequence → Multi-Resolution Subbands)

We apply the **Stationary Wavelet Packet Transform (SWPT)** with a **full-tree** decomposition to sequence features, producing multiple subband representations that are **equal-length** and **shift-invariant**. Compared to downsampling-based wavelet packet transforms, SWPT preserves sequence length and improves alignment stability, which is well-suited for position-aligned modeling in sequential recommendation.

To mitigate boundary artifacts, we use:
- **symmetric extension**, and
- **learnable boundary tokens** (to absorb spurious boundary signals).

Each subband corresponds to a specific temporal scale / frequency component and serves as input for subband-level modeling.

### 2) Per-Subband Sequence Aggregation (Per-Subband Attention)

For each subband, we adopt a lightweight **additive attention** module to aggregate the time steps into a subband-specific user representation. Different subbands may emphasize different key positions; per-subband attention enables scale-specific selection of informative segments.

### 3) Band-Consistent Graph Propagation (Per-Subband Chebyshev Propagation)

We perform graph enhancement on the user–item interaction graph using **Chebyshev polynomial spectral propagation**. Importantly, propagation is **band-consistent**: each subband is associated with its own propagation branch, allowing graph information to be injected and modulated independently across different sequence scales.

---

## Adaptive Fusion (Energy–Spectral-Flatness Gated Fusion)

Subband-level representations are fused into the final user embedding. WPGRec uses **energy** and **spectral flatness** as gating signals to modulate each subband’s contribution:
- energy measures signal strength,
- spectral flatness reflects how “noise-like” (uniform) versus structured a subband is.

This gated fusion allows the model to emphasize the most informative scales adaptively across users and datasets.  
For items, we use **uniform averaging** for fusion by default (consistent with the paper setting).

---

## Contributions

Our main contributions can be summarized as follows:

1. **A multi-resolution sequential modeling framework**: we use full-tree SWPT to explicitly decompose preference signals into multiple subbands and learn representations at the subband level.  
2. **A subband-level graph enhancement mechanism**: we introduce band-consistent Chebyshev spectral propagation on the user–item graph to inject structural information independently at each scale.  
3. **Interpretable gated fusion**: we design an energy–spectral-flatness gating scheme to enable controllable and diagnosable subband fusion.

---

## Features

- Full-tree **SWPT**: equal-length, shift-invariant subbands
- Symmetric extension + **learnable boundary tokens**
- **Per-subband additive attention** for sequence aggregation
- **Per-subband Chebyshev** spectral propagation for graph enhancement
- Energy + spectral flatness **gated fusion** (user side), uniform averaging for item fusion
- Full-ranking evaluation: HR@K / NDCG@K, excluding training-history items from candidates

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .