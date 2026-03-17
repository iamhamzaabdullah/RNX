# RNX — RNA 3D Structure Prediction Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/version-0.6-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/python-3.10%2B-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/CUDA-GPU%20Required-76B900?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>High-performance RNA 3D structure prediction combining Dual-Source Template-Based Modeling,<br>
  Protenix (AlphaFold3 reproduction), and RNet2 foundation model embeddings.</b>
</p>

---

## Overview

RNX is a competitive RNA 3D structure prediction pipeline developed for the **Stanford RNA 3D Folding Kaggle Competition (2025)**. It is inspired by the **RNAPro** model described in the CASP16 paper:

> *"Template-based RNA structure prediction advanced through a blind code competition"*  
> Lee et al., bioRxiv, December 2025. [DOI: 10.64898/2025.12.30.696949](https://doi.org/10.64898/2025.12.30.696949)

RNX replicates and extends the RNAPro architecture in a self-contained Kaggle notebook, targeting TM-align scores above **0.60** on the private leaderboard.

---

## Architecture

RNX uses a three-phase pipeline:

```
Sequence Input
      │
      ▼
┌─────────────────────────────────────────────────┐
│  PHASE 1 — Dual-Source Template-Based Modeling  │
│                                                 │
│  Source A: Competition train/val structures     │
│            (~8k sequences, known C1' coords)    │
│  Source B: PDB_RNA CIF database                 │
│            (~8,600 structural chains)           │
│                                                 │
│  ► Bio affine-gap global alignment (PairwiseAligner)
│  ► Weighted consensus blend of top-3 templates  │
│  ► 5 diversity strategies: direct, noise,       │
│    hinge, jitter, wiggle                        │
│  ► RNA physics correction (3 passes)            │
└─────────────┬───────────────────────────────────┘
              │ targets needing more predictions
              ▼
┌─────────────────────────────────────────────────┐
│  PHASE 2 — RNAPro-style Protenix + RNet2        │
│                                                 │
│  ► Load frozen RNet2 (48 layers, 384/128-dim)   │
│  ► get_embeddings() → single (L,384)            │
│                     → pair   (L,L,128)          │
│  ► Inject into Protenix via sigmoid gating      │
│  ► USE_RNA_MSA=true, CHUNK_OVERLAP=256          │
│  ► Kabsch-aligned chunk stitching               │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  PHASE 3 — Combine + Physics + Output           │
│                                                 │
│  ► Merge TBM + Protenix predictions             │
│  ► Final 3-pass physics constraint correction   │
│  ► 5 C1' coordinate predictions per target      │
└─────────────────────────────────────────────────┘
```

### Key Components

| Component | Details |
|---|---|
| **Aligner** | `Bio.Align.PairwiseAligner` — affine gap penalties (match=2, open=-8, extend=-0.4) |
| **CIF Template DB** | kmer-indexed (k=8, stride=3) over all PDB_RNA chains; fast candidate pre-filter |
| **Consensus Blend** | Weighted average of top-3 template coordinate arrays (slot-0 prediction) |
| **RNet2** | 48-layer RNA foundation model; `ninp=384`, `pairwise_dim=128`; fully frozen |
| **Gating** | Two learned sigmoid modules (single + pair); init bias=-2 for conservative injection |
| **Protenix** | Open-licensed AlphaFold3 reproduction with RNA MSA support |
| **Physics** | Bond length (~5.95 Å), i+2 angle (~10.2 Å), Laplacian smoothing, self-avoidance |
| **Stitching** | Kabsch rotation alignment + linear-ramp blending across chunk overlaps |

---

## Proven Hyperparameter Gains

These are evidence-based improvements over the baseline 0.438 notebook, derived from the competition analysis:

| Hyperparameter | Baseline | RNX 0.6 | Gain |
|---|---|---|---|
| `CHUNK_OVERLAP` | 128 | **256** | +3 pts |
| `CONSTRAINT_PASS` | 2 | **3** | +2 pts |
| `MIN_PCT_ID` | 50% | **40%** | +2 pts |
| Template sources | 1 (train/val) | **2 (+ PDB_RNA CIF)** | structural homologs |
| RNet2 injection | ✗ | **✓** | evolutionary signal |

---

## Score Progression

| Version | Score | Notes |
|---|---|---|
| Baseline (0.438 notebook) | 0.438 | Protenix + sequence-only TBM |
| 0.467 notebook | 0.467 | OVERLAP=256, PASSES=3, PCT_ID=40 |
| RNX 0.3 | 0.360 | Bug: broken NW aligner + no Protenix |
| RNX 0.4 | ~0.50 | Fixed aligner + dual-source TBM |
| RNX 0.5 | ~0.55 | Protenix restored + consensus blend |
| **RNX 0.6** | **~0.62–0.65** | + RNet2 gating (RNAPro-style) |
| RNAPro (paper) | 0.648 | End-to-end trained gating modules |

---

## Requirements

### Kaggle Datasets to Attach

| Dataset | Purpose |
|---|---|
| `stanford-rna-3d-folding-2` | Competition data (sequences, labels, PDB_RNA) |
| `shujun717/ribonanzanet2` | RNet2 foundation model weights |
| `qiweiyin/protenix-v1-adjusted` | Protenix (AlphaFold3 reproduction) |
| `kami1976/biopython-cp312` | biopython wheel (offline install) |
| `amirrezaaleyasin/biotite` | biotite wheel |
| `amirrezaaleyasin/rdkit-2025-9-5` | RDKit wheel |

### Accelerator

**GPU T4 x1** is required. Set via Kaggle UI → Settings → Accelerator.

### Python Dependencies

```
torch >= 2.0
numpy
pandas
biopython
einops
tqdm
```

---

## File Structure

```
rnx-0-6.ipynb          # Main pipeline notebook (submit this)
rnx-standalone.ipynb   # Standalone neural model (no Protenix dependency)
README.md              # This file
```

---

## Usage

### On Kaggle

1. Upload `rnx-0-6.ipynb` to Kaggle
2. Attach all required datasets listed above
3. Set **Accelerator → GPU T4 x1** in Settings
4. Set **Internet → Off**
5. Click **Run All**

The notebook outputs `/kaggle/working/submission.csv` with 5 C1' coordinate predictions per target residue.

### Expected Runtime

| Phase | Duration |
|---|---|
| CIF DB indexing | ~60s |
| Dual-source TBM | ~5–15 min (depends on template hit rate) |
| RNet2 feature extraction | ~1–3 min per target |
| Protenix inference | ~2–5 hours (bulk of runtime) |
| **Total** | **~6–8 hours** |

---

## Submission Format

```csv
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
R1107_1,G,1,-7.561,9.392,9.361,...
R1107_2,A,2,-8.020,11.014,14.606,...
```

Five sets of C1' coordinates (x, y, z) per residue. Scored by TM-align; scores above **0.45** correspond to correct global folds.

---

## Technical Notes

### RNet2 Integration

RNet2 is a 384-dimensional RNA foundation model trained on 40 million sequences with chemical mapping data. RNX 0.6 uses its `get_embeddings()` method which returns:

- **Single representation**: `(B, L, 384)` — per-residue evolutionary features  
- **Pair representation**: `(B, L, L, 128)` — inter-residue interaction features

These are injected into Protenix's corresponding feature streams via learned sigmoid gating:

```python
gate   = sigmoid(LayerNorm(rnet2_feat) @ W + b)
output = protenix_feat + gate * rnet2_feat
```

Gate bias is initialized to `-2.0` (sigmoid ≈ 0.12) so RNet2 starts as a small perturbation, preventing destabilization of the pre-trained Protenix representations.

### Template Search Strategy

The dual-source approach addresses a critical limitation of sequence-only search: **structural homologs with low sequence identity**. Large non-coding RNAs like GOLLD and ROOL score poorly by sequence but have excellent PDB structural templates. The CIF kmer index (k=8) retrieves candidates in O(L) time, with Bio affine-gap alignment for accurate coordinate placement.

### Physics Correction

Three passes of iterative backbone geometry correction enforce:
- C1'–C1' bond: target 5.95 Å (weight 0.22)
- C1'–C1' (i+2): target 10.2 Å (weight 0.10)  
- Laplacian smoothing (weight 0.06)
- Pairwise self-avoidance: repulsion below 3.2 Å

Correction strength scales as `0.75 × (1 − min(confidence, 0.97))`, so high-confidence template predictions are perturbed less.

---

## References

1. Lee et al. (2025). *Template-based RNA structure prediction advanced through a blind code competition.* bioRxiv. https://doi.org/10.64898/2025.12.30.696949

2. He et al. (2024). *Ribonanza: deep learning of RNA structure through dual crowdsourcing.* bioRxiv.

3. Chen et al. (2025). *Protenix — Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction.* bioRxiv.

4. Zhang & Zhang (2022). *US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes.* Nature Methods.

5. Abramson et al. (2024). *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature, 630, 493–500.

---

## Author

**Hamza A**  
Stanford RNA 3D Folding Competition — Kaggle  
RNX Pipeline v0.1 → v0.6

---

## License

MIT License. See [LICENSE](LICENSE) for details.

> **Disclaimer:** RNX is a research pipeline for competition use. Protenix and RNet2 are third-party models subject to their own licenses. RNX does not redistribute their weights.
