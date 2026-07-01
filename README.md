# RLE-Feature Classifier: Why Run-Length Encoding Beats Raw Pixels on Efficiency

**A compact, structure-aware alternative to feeding raw pixels into a CNN.** Instead of
handing a network the full 2D pixel grid and letting it learn spatial filters from
scratch, this project compresses each grayscale image into two short run-length-encoded
sequences first — and shows that a small model built on top of that compressed
representation gets *close to* CNN-level accuracy at a fraction of the parameters and
compute.

This README makes the efficiency case: what RLE saves, why it saves it, and where it
already wins outright.

---

## Table of Contents

- [The Pitch](#the-pitch)
- [Why Raw Pixels Are Wasteful](#why-raw-pixels-are-wasteful)
- [How RLE Compresses an Image](#how-rle-compresses-an-image)
- [Architecture](#architecture)
  - [RLEClassifier](#rleclassifier)
  - [CrossAttention1D](#crossattention1d)
  - [What It's Compared Against](#what-its-compared-against)
- [The Efficiency Numbers](#the-efficiency-numbers)
- [Where RLE Wins Outright](#where-rle-wins-outright)
- [The Honest Accuracy Picture](#the-honest-accuracy-picture)
- [Tuning the Encoding: MAX_RUNS and THRESH_K](#tuning-the-encoding-max_runs-and-thresh_k)
- [Why This Matters](#why-this-matters)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Notebook Map](#notebook-map)
- [Reproducing](#reproducing)

---

## The Pitch

A raw grayscale image is mostly redundancy. Neighboring pixels repeat the same value
over and over; a CNN spends a huge number of parameters and FLOPs re-discovering that
redundancy every forward pass just to compress it back down internally. **Run-length
encoding does that compression up front, for free, before a single weight is learned.**

The result, measured across four real datasets with 5-fold cross-validation against a
parameter-matched raw-pixel CNN baseline:

- **~1.7–1.8× fewer parameters**, consistently, across every dataset tested
- **~19× fewer MACs/FLOPs** — the model does dramatically less arithmetic per forward pass
- **Outright accuracy win on the smallest dataset** (+13.8 accuracy points over the CNN)
- **Statistical tie on a mid-size 4-class MRI task**, at little more than half the parameter budget

That's the core proof-of-concept claim: **a compressed 1D structural summary of an
image can substitute for raw pixels**, and the efficiency gap this opens up is large
and consistent, even where the CNN still holds a small accuracy edge.

---

## Why Raw Pixels Are Wasteful

A CNN operating on raw pixels has to learn, from scratch, that:

- Long runs of similar-valued pixels are one "thing," not many independent signals
- Spatial position and local intensity together define structure (edges, blobs, regions)
- Most of the 2D grid it's looking at is low-information filler around the actual
  diagnostic or discriminative structure

RLE encodes exactly this prior *directly into the input representation*, before
training even starts. Every row and column is reduced to "where does a bright region
start, how long does it run, and how bright is it, on average" — up to a handful of
runs per row/column. That's a massive dimensionality drop with a well-defined,
interpretable meaning, and it means the downstream model doesn't have to spend
capacity rediscovering it.

---

## How RLE Compresses an Image

```
grayscale image (H×W)
        │
        ▼
  threshold → binary mask        (mean + THRESH_K · std, adaptive per image)
        │
   ┌────┴────┐
   ▼         ▼
row-wise   col-wise
RLE scan   RLE scan
   │         │
   ▼         ▼
(start, length, mean-intensity) × MAX_RUNS   per row / per column
   │         │
   ▼         ▼
flatten → row_vector   (CROP_ROWS  · MAX_RUNS · 3)
flatten → col_vector   (IMAGE_SIZE · MAX_RUNS · 3)
```

For every row (and, separately, every column via a transposed pass), the binarized
image is scanned for runs of "on" pixels. Each run is reduced to three numbers:

| Field | Meaning | Normalization |
|---|---|---|
| `start` | Where the run begins | ÷ image width |
| `length` | How long the run is | ÷ image width |
| `mean_intensity` | Average raw pixel value over the run | ÷ 255 |

At the default `MAX_RUNS = 4`, a 64×64 image collapses from **4,096 raw pixel values**
down to a **1,536-value structured vector** (768 row + 768 col) — already a ~2.7×
reduction before a single learned parameter is involved, and this compressed vector is
*all* the downstream model ever sees.

Two implementations exist and are auto-selected at runtime: a parallel Numba-JIT
scanner for speed, and a NumPy fallback (vectorized via boundary-diff detection) if
`numba` isn't installed — functionally identical, just slower.

A dedicated reconstruction-check cell in the notebook (`check_single_image_reconstruction`)
visually confirms the encoding faithfully captures the binarized image's structure
before any model ever touches it.

---

## Architecture

### RLEClassifier

Two independent, lightweight 1D-CNN branches — one for the row profile, one for the
column profile — fused with a small cross-attention module.

**Per-branch (row and column share this structure, separate weights):**

| Stage | Layer | Channels | Kernel / Stride |
|---|---|---|---|
| 1 | `Conv1d` + BN + ReLU + Dropout | 1 → 32 | k=3, s=3 |
| 2 | `Conv1d` + BN + ReLU + Dropout | 32 → 64 | k=4, s=4 |
| 3 | `Conv1d` + BN + ReLU + Dropout | 64 → 64 | k=3, s=2 |
| — | `AdaptiveAvgPool1d(2)` | — | pools to a fixed length |

Each branch collapses to a `128`-dim vector (`64 channels × pool_out=2`). The two
branch vectors are fused via `CrossAttention1D`, concatenated (`fused_dim = 256`), and
passed through a small FC head (`256 → 64 → 32 → num_classes`).

**Total parameters: ~69K–97K**, depending on resolution and class count — roughly
**half** the size of the comparison CNN at every resolution tested.

### CrossAttention1D

A lightweight **gated cross-attention** — not full scaled dot-product attention —
where each branch gates the other:

```python
def forward(self, row, col):
    row_out = row * self.gate_row(col)   # column context gates the row signal
    col_out = col * self.gate_col(row)   # row context gates the column signal
    return row_out, col_out
```

Each gate is a tiny bottleneck MLP (`Linear → ReLU → Linear → Sigmoid`). This lets
horizontal and vertical structure inform each other — useful when a discriminative
shape (e.g. a tumor's silhouette) only becomes clear once row and column information
are considered jointly — while staying far cheaper than standard multi-head attention.

### What It's Compared Against

To make the efficiency claim rigorous rather than anecdotal, RLEClassifier is
benchmarked against **FairCNNBaseline**: a raw-pixel 2D CNN deliberately built with
the *same* `fused_dim` (256) and the *same* FC head shape, so the only real difference
between the two models is **what they're allowed to see** — compressed RLE structure,
vs. every raw pixel. That's what makes the parameter and FLOP gap below a fair,
apples-to-apples efficiency result rather than an artifact of an unfairly small
comparison model.

---

## The Efficiency Numbers

This is the headline result of the whole project.

| Metric | RLEClassifier | FairCNNBaseline | RLE's advantage |
|---|---|---|---|
| Parameters | ~96–97K (128×128) / 69K (64×64) | ~172K (128×128) / 116K (64×64) | **~1.7–1.8× fewer** |
| MACs / FLOPs | ~6.9M | ~133M | **~19.5× fewer** |
| Memory footprint (float32) | ~0.27 MB | ~0.44–0.66 MB | roughly half |

This holds **consistently across every single dataset tested** — chest X-ray, brain
MRI, breast ultrasound, and handwritten character recognition. It's not a lucky result
on one dataset; it's a structural property of feeding a network a compressed 1D
representation instead of a full 2D grid.

**On GPU wall-clock time**, the CNN currently comes out ahead (2.25ms vs 3.41ms
inference; 19.24s vs 21.45s per training fold) — 2D convolutions on regular pixel
grids are exactly the workload GPUs are built to parallelize, so 19× fewer FLOPs
doesn't yet translate into 19× less wall-clock time on this hardware. This is a real
and well-known tension in ML systems (compute-efficient ≠ hardware-efficient), and it's
flagged here rather than glossed over — see [Limitations](#limitations). It also
means RLE's efficiency case is currently strongest exactly where FLOPs and parameter
count matter most directly: **storage-constrained, memory-constrained, or CPU/embedded
deployment**, rather than GPU throughput.

---

## Where RLE Wins Outright

| Dataset | N (train) | Classes | RLE Acc | CNN Acc | Gap |
|---|---|---|---|---|---|
| **Ultrasound (breast)** | 956 | 3 | **0.7605** | 0.6228 | **+13.8 pts — RLE wins clearly** |
| Brain Tumor (MRI) | 5,600 | 4 | 0.9416 | 0.9420 | ~0 pts — statistical tie |

On the smallest dataset tested, RLE doesn't just keep pace with the CNN — it wins
**every single fold** on accuracy and F1, by a wide margin (+13.8 accuracy points,
+11.3 macro-F1 points, well outside the fold-to-fold noise). The CNN, with nearly
twice the parameters and no built-in structural prior, visibly underfits with so
little data (training accuracy stalls near 70% even after 100 epochs). RLE's
fixed encoding acts as a **built-in regularizer** — it doesn't need to learn what
"image structure" means from scratch, so it needs far less data to find a good
decision boundary.

On the 4-class brain tumor MRI task, RLE **ties** the CNN on every CV metric
(differences of 0.0002–00004, smaller than the fold-to-fold standard deviation) while
using **~56% of the parameters**. Matching a raw-pixel CNN at roughly half the budget
is exactly the efficiency case this project sets out to make.

---

## The Honest Accuracy Picture

For completeness — and because a proof-of-concept is only credible if it shows its
limits — the CNN does pull ahead on the two datasets with more training data and
fine-local-texture-dependent signal:

| Dataset | N (train) | RLE Acc | CNN Acc | Gap |
|---|---|---|---|---|
| Pneumonia (chest X-ray) | 5,232 | 0.9606 | 0.9801 | CNN +1.95 pts |
| KMNIST (handwritten chars) | 60,000 | 0.9798 | 0.9934 | CNN +1.36 pts |

Note the size of these gaps, though: **1–2 accuracy points**, in exchange for
**roughly half the parameters and ~19× fewer FLOPs**. RLE at 97.98% accuracy on
KMNIST with 96,906 parameters is a strong result in absolute terms, even in the one
comparison it "loses." The pattern across all four datasets suggests RLE's advantage is
largest in exactly the regime most valuable in practice — **small, resource-constrained
datasets**, like most real-world clinical imaging — while its cost is modest even
where the CNN wins.

---

## Tuning the Encoding: MAX_RUNS and THRESH_K

The two knobs that control the fidelity/compression trade-off of the RLE
representation are configurable and **have not yet been swept or tuned**:

- **`MAX_RUNS`** — how many runs are kept per row/column before truncation (default:
  `4`). Raising it captures more fine-grained structure per row at the cost of a
  larger feature vector (and slightly more downstream compute); lowering it compresses
  further but risks losing runs in busy/high-contrast rows. All results in this README
  used `MAX_RUNS = 4`.
- **`THRESH_K`** — the multiplier on the per-image standard deviation used to set the
  binarization threshold (`threshold = mean + THRESH_K · std`, default: `0.2`). This
  controls how aggressively pixels are classified as "on" vs. "off" before run
  detection, and is currently set per-project rather than tuned per-dataset.

Because both values were chosen once and held fixed across all four datasets for a
fair comparison, **the efficiency numbers above are a lower bound, not a ceiling** — a
per-dataset sweep of `MAX_RUNS`/`THRESH_K` could plausibly close some or all of the
remaining accuracy gap on Pneumonia/KMNIST, or compress the representation even
further on datasets where 4 runs/row is already more than needed. This is one of the
most direct, low-effort next steps for improving the RLE side of the comparison
further (see [Roadmap](#roadmap)).

---

## Why This Matters

Fewer parameters and dramatically fewer FLOPs are not just a nice-to-have — they map
directly onto real deployment constraints:

- **Storage / memory footprint** — smaller models ship in smaller binaries and fit in
  tighter memory budgets (edge devices, browser-based inference, mobile).
- **Low-data regimes** — RLE's built-in structural prior is a free regularizer exactly
  when data is scarce, which is the common case in specialized domains like medical
  imaging.
- **Theoretical compute cost** — ~19× fewer MACs is a genuinely large reduction in the
  raw arithmetic a deployment target has to be provisioned for, even before any
  hardware-specific optimization work is done.
- **A different scaling story** — the CNN's advantage grows (modestly) as data grows;
  RLE's advantage grows as data shrinks. For any team without millions of labeled
  images, that's the more relevant end of the curve.

This is why the project frames RLE as **worth taking seriously as a lightweight
alternative representation** — not a universal pixel replacement, but a structurally
different, far cheaper option that already wins outright in the low-data regime and
stays within a couple of accuracy points everywhere else.

---

## Limitations

Being upfront about what hasn't been shown yet keeps the efficiency claim credible:

- **GPU wall-clock time currently favors the CNN**, despite RLE's much lower FLOP
  count — 2D convolutions parallelize better on GPU than RLE's 1D-conv +
  cross-attention pipeline. This gap hasn't been root-caused yet (kernel launch
  overhead? sequence length? the attention module specifically?), and a CPU-only
  benchmark — where GPU parallelism advantages disappear — hasn't been run yet either,
  even though it's the setting most likely to convert RLE's FLOP advantage into an
  actual wall-clock win.
- **Not a rigorously tuned model.** Architecture and hyperparameters (including
  `MAX_RUNS` and `THRESH_K`, see above) were fixed for a fair head-to-head comparison,
  not optimized for best possible RLE performance.
- **Grayscale/B&W only** — the encoding assumes a single intensity channel and hasn't
  been extended to color.
- **Tested at small-to-medium scale only** (956–60,000 images); behavior at
  million-image scale is untested.
- **RLE's win is data-size-dependent** — it's decisive on the smallest dataset, a tie
  on a mid-size one, and a modest loss on the two largest/most texture-dependent ones.

---

## Roadmap

1. **Sweep `MAX_RUNS` and `THRESH_K` per dataset** — the most direct lever for
   closing the remaining accuracy gap without adding parameters.
2. **CPU-only benchmarking** — the most likely place for RLE's FLOP advantage to
   convert into an actual, provable wall-clock win.
3. **RLE-specific architecture search**, instead of reusing a shape mirrored from the
   CNN baseline for fair comparison.
4. **Color (RGB) extension** — per-channel binarize + RLE + fuse, testing whether the
   efficiency case survives tripling the pipeline for 3 channels.
5. **N-sweep on a single fixed task** (e.g. subsample Brain Tumor or Pneumonia to
   ~1,000 images) to cleanly isolate whether RLE's low-data edge is driven by sample
   size, task structure, or both — currently confounded across the four datasets.
6. **Scale testing** well past 60,000 images to see whether the CNN's edge widens,
   narrows, or disappears with more data.

---

## Notebook Map

| Stage | Cell(s) | Contents |
|---|---|---|
| Setup | 1 | Imports, seeding |
| Data loading | 2 | Dataset-specific image loading |
| Configuration | 4 | `IMAGE_SIZE`, `MAX_RUNS`, `THRESH_K`, model/training hyperparameters |
| RLE encoding | 5 | `vectorized_rle` (Numba + NumPy fallback), `_rle_from_array` |
| Reconstruction check | 4B | Visual sanity check of the RLE encoding |
| Feature extraction | 6–7 | `extract_sequential`, `normalize_features` |
| Feature diagnostics | — | `assess_rle_features` (class separability, variance) |
| RLE architecture | 8 | `FocalLoss`, `RLEDataset`, `CrossAttention1D`, `RLEClassifier` |
| Loss builder | 8.5 | `compute_loss_fn` |
| Fast training loop | 8.7 | Direct-GPU-slicing `train_epoch_fast` / `evaluate_fast` |
| RLE cross-validation | 8.8 | 5-fold CV for `RLEClassifier` |
| Optional XGBoost head | 10B/10C | Classical ML on extracted conv features |
| CNN baseline | 8B | `FairCNNBaseline`, `CNNDataset` |
| CNN training | 9B | GPU-preloaded fast training loop |
| CNN cross-validation | 9C | 5-fold CV for `FairCNNBaseline` + comparison table |
| Model inspector | 8C | Per-layer parameter counts for both models |

---

## Reproducing

**Dependencies:** `torch`, `numpy`, `opencv-python`, `Pillow`, `scikit-learn`, `tqdm`;
optional `numba` (falls back to NumPy automatically) and `xgboost` (only for the
optional feature-extraction path).

**To run on a new dataset:**

1. Point the data-loading cell at your image folders (grayscale, one subfolder per
   class).
2. Set `IMAGE_SIZE`, `NUM_CLASSES`, `CLASS_NAMES` — and consider sweeping `MAX_RUNS`
   / `THRESH_K` for your specific image statistics rather than reusing the defaults.
3. Run the RLE encoding + feature extraction cells to build cached feature arrays.
4. Run Cell 8 (architecture) → Cell 8.8 (RLE 5-fold CV).
5. Run Cell 8B (`FairCNNBaseline`) → Cell 9C (CNN 5-fold CV) for the efficiency/accuracy
   comparison.
6. Cell 8C prints a side-by-side parameter and memory breakdown for both models.

All randomness is seeded (`SEED = 42`); `cudnn.benchmark = True` is enabled for speed,
so exact fold-level numbers may vary slightly run-to-run.
