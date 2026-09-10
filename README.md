# WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation

WPGRec (Wavelet Packet Guided Graph Enhanced Sequential Recommendation) is a multi-resolution framework for sequential recommendation. It combines time-frequency decomposition, scale-consistent graph propagation, and adaptive subband fusion to model user interests at different temporal scales.

## Background

Sequential recommendation aims to predict a user’s next interaction based on their historical interaction sequence.

Real-world user behavior sequences usually contain preference signals at multiple temporal scales, including short-term interest fluctuations, periodic behavior patterns, and relatively stable long-term preferences. Traditional sequential models generally encode these signals within a single representation space, which may make it difficult to distinguish interest variations across different temporal scales.

Meanwhile, user-item interactions naturally form a user-item graph containing high-order collaborative information beyond individual behavior sequences. Therefore, an effective sequential recommendation model should not only capture user interests at multiple temporal scales but also incorporate graph-structured information without disrupting the temporal structure of the sequence.

From a multi-resolution signal modeling perspective, WPGRec decomposes user behavior sequences into multiple equal-length subbands, performs temporal aggregation and graph propagation independently within each subband, and finally combines the resulting representations through an adaptive gating mechanism.

## Model Architecture

WPGRec consists of the following components.

### 1. Full-Tree Stationary Wavelet Packet Decomposition

WPGRec first applies boundary processing to the embedding representation of each user behavior sequence. Symmetric extension and learnable boundary tokens are employed to mitigate boundary distortions caused by finite-length sequences during wavelet decomposition.

The model then applies a full-tree Stationary Wavelet Packet Transform (SWPT) along the temporal dimension.

Unlike conventional wavelet packet transforms involving downsampling, SWPT produces subbands with equal sequence lengths and shift-invariant properties. Consequently, temporal positions remain aligned across different resolutions, providing a stable foundation for subsequent subband-level graph propagation and fusion.

Different subbands represent user interests at different temporal scales and frequency ranges, including:

- Relatively stable long-term interests;
- Behavioral variations over intermediate time ranges;
- Local and short-term interest fluctuations.

### 2. Per-Subband Sequence Aggregation

For each wavelet subband, WPGRec employs an independent additive attention mechanism to aggregate information across temporal positions.

The attention module assigns weights according to the importance of different positions and aggregates the sequential features within each subband into a scale-specific user representation. Different subbands can focus on different informative interactions, enabling the model to learn user representations with distinct temporal characteristics.

### 3. Band-Consistent Graph Propagation

WPGRec constructs a bipartite user-item graph from the training interactions and performs graph propagation independently within each subband.

The graph propagation module employs Chebyshev-polynomial-based spectral graph filtering to inject high-order collaborative information into user and item representations at different scales.

Unlike methods that first mix all temporal scales and then introduce graph information uniformly, WPGRec performs graph propagation independently within each subband. This keeps the temporal decomposition scale consistent with the graph enhancement scale and reduces interference among different frequency components.

### 4. Energy and Spectral-Flatness-Aware Gated Fusion

After subband-level graph propagation, the representations from multiple scales are combined into the final user representation.

WPGRec uses subband energy and the Spectral Flatness Measure (SFM) as gating signals:

- Subband energy describes the signal strength of each frequency component;
- Spectral flatness indicates whether a subband is more structured or noise-like.

A learnable gating network calculates the fusion weight of each subband. This mechanism assigns larger weights to informative subbands while reducing the influence of noise-like components on the final prediction.

Adaptive weighted fusion is used on the user side, while item representations from different subbands are combined through uniform averaging by default.

### 5. Prediction and Optimization

After cross-subband fusion, the model obtains the final user and item representations and predicts the user’s next interaction based on their matching scores.

The model is trained using full-softmax cross-entropy without sampled negatives. During evaluation, each ground-truth item is ranked against the complete candidate item set, excluding items already observed in the user’s training history.

The primary evaluation metrics include:

- HR@10;
- HR@20;
- NDCG@10;
- NDCG@20.

A concise mapping between the model components and the corresponding equations in the paper is provided in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Contributions

1. We propose a multi-resolution framework for sequential recommendation that explicitly decomposes user interests at different temporal scales using a full-tree Stationary Wavelet Packet Transform.

2. We introduce a band-consistent graph propagation mechanism that independently incorporates high-order collaborative information from the user-item graph at different temporal scales.

3. We design an energy and spectral-flatness-aware gated fusion mechanism that adaptively selects informative subbands and suppresses noise-like components.

## Data Preprocessing

The data preprocessing utility accepts interaction records containing user IDs, item IDs, and timestamps. It performs ID remapping and chronological sorting of the interaction records.

The expected input format is:

```text
user_id    item_id    timestamp
```

Install the data preprocessing utility with:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e .
```

Run preprocessing with:

```bash
wpgrec-preprocess \
  --in_path path/to/interactions.tsv \
  --out_dir path/to/processed_data \
  --u_col 0 \
  --i_col 1 \
  --t_col 2
```

If the input file contains a header, add:

```bash
--has_header
```

The processed interaction records and corresponding dataset statistics will be saved in the specified output directory.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{liu2026wpgrec,
  title     = {WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation},
  author    = {Liu, Peilin and Ji, Zhiquan and Yan, Gang},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year      = {2026},
  doi       = {10.1145/3805712.3809907}
}
```

## License

This project is released under the MIT License.

> **Note:** The complete model implementation and detailed reproduction instructions are currently being organized and will be released in this repository soon.
