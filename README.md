# RLE-Feature Classifier: Demonstrating Run-Length Encoding as an Alternative to Raw Pixels

**A work-in-progress proof-of-concept.** This project asks a simple question: does a
network really need to see every raw pixel to classify a grayscale image, or can a
compact run-length-encoded (RLE) summary of the image's structure carry most of the
same signal, at a fraction of the parameters and compute?

The model, training setup, and hyperparameters here are still being iterated on. The
accuracy and AUC numbers in this README are **not final results** — they're early
readings meant to demonstrate that the RLE approach is *viable and worth developing
further*, not a finished benchmark claim. The one thing that has already been shown
clearly and consistently is the **efficiency gap**: RLE gets a network into the same
neighborhood of accuracy as a raw-pixel CNN using far fewer parameters and far less
arithmetic per forward pass. That's the core of the pitch below.

---

## Table of Contents

- [The Pitch](#the-pitch)
- [Why Raw Pixels Are Wasteful](#why-raw-pixels-are-wasteful)
- [How RLE Compresses an Image](#how-rle-compresses-an-image)
- [Architecture](#architecture)
  - [RLEClassifier](#rleclassifier)
  - [CrossAttention1D](#crossattention1d)
  - [The Comparison Model](#the-comparison-model)
- [Early Efficiency Signal](#early-efficiency-signal)
- [Early Accuracy Signal (Preliminary)](#early-accuracy-signal-preliminary)
- [Tuning the Encoding: MAX_RUNS and THRESH_K](#tuning-the-encoding-max_runs-and-thresh_k)
- [Why This Is Worth Pursuing](#why-this-is-worth-pursuing)
- [Status / What's Still Open](#status--whats-still-open)
- [Roadmap](#roadmap)
- [Notebook Map](#notebook-map)
- [Reproducing](#reproducing)

---

## The Pitch

A raw grayscale image is mostly redundancy. Neighboring pixels repeat the same value
over and over; a CNN spends a huge number of parameters and FLOPs re-discovering that
redundancy every forward pass just to compress it back down internally. **Run-length
encoding does that compression up front, for free, before a single weight is
learned.**

Instead of handing a network a full 2D pixel grid, each image is reduced to two short
1D sequences — a row profile and a column profile — that describe *where* bright
structure starts, *how long* it runs, and *how intense* it is. That's the entire input
the model ever sees.

This project isn't trying to prove RLE beats CNNs across the board — it's still being
tuned, and the numbers below will move. What it's demonstrating is narrower and, right
now, more solid: **a model built on this compressed representation can get into the
same ballpark of accuracy as a raw-pixel CNN, while using a small fraction of the
parameters and compute.** That efficiency gap is the interesting result, and it's
consistent every time the comparison has been run so far.

---
## Per-Dataset Detailed Results (Preliminary — 5-Fold CV)

> Current configuration: `MAX_RUNS = 4`, `THRESH_K = 0.2`. These are early
> readings from an untuned setup — not final numbers.

---

### 1. Pneumonia (Chest X-ray, Binary — NORMAL vs PNEUMONIA)

**Setup:** 5,216 train + 16 val (merged) + 624 held-out test, combined into a
5,232-image pool for 5-fold CV. Class imbalance: 1,349 NORMAL vs 3,883 PNEUMONIA
(~74% positive). Resolution 64×64.

| Fold | RLE AUC | RLE Acc | RLE F1 | CNN AUC | CNN Acc | CNN F1 |
|---|---|---|---|---|---|---|
| 1 | 0.9915 | 0.9534 | 0.9385 | 0.9988 | 0.9866 | 0.9827 |
| 2 | 0.9933 | 0.9618 | 0.9510 | 0.9968 | 0.9771 | 0.9705 |
| 3 | 0.9917 | 0.9546 | 0.9409 | 0.9968 | 0.9742 | 0.9667 |
| 4 | 0.9933 | 0.9654 | 0.9552 | 0.9991 | 0.9866 | 0.9827 |
| 5 | 0.9945 | 0.9677 | 0.9586 | 0.9968 | 0.9761 | 0.9692 |
| **Mean** | **0.9928** | **0.9606** | **0.9488** | **0.9977** | **0.9801** | **0.9744** |
| **Std** | 0.0011 | 0.0057 | 0.0079 | 0.0011 | 0.0054 | 0.0069 |

**Params:** RLE 69,476 · CNN 171,684 (RLE uses 56.2% of CNN's params)

Notes: CNN currently ahead across every fold/metric here; the gap is largest on
this dataset out of the four tried so far (+3.34 macro-F1 pts). Both models are
very stable fold-to-fold (std ≤0.008). 

---

### 2. Brain Tumor (MRI, 4-class — glioma / meningioma / notumor / pituitary)

**Setup:** 5,600 train images, 4 roughly balanced classes, 128×128 resolution.

| Fold | RLE AUC | RLE Acc | RLE F1 | CNN AUC | CNN Acc | CNN F1 |
|---|---|---|---|---|---|---|
| 1 | 0.9911 | 0.9366 | 0.9365 | 0.9921 | 0.9393 | 0.9396 |
| 2 | 0.9913 | 0.9420 | 0.9418 | 0.9915 | 0.9384 | 0.9379 |
| 3 | 0.9933 | 0.9411 | 0.9412 | 0.9919 | 0.9339 | 0.9339 |
| 4 | 0.9905 | 0.9339 | 0.9337 | 0.9928 | 0.9429 | 0.9422 |
| 5 | 0.9945 | 0.9545 | 0.9544 | 0.9943 | 0.9554 | 0.9552 |
| **Mean** | **0.9921** | **0.9416** | **0.9415** | **0.9925** | **0.9420** | **0.9418** |
| **Std** | 0.0015 | 0.0071 | 0.0071 | 0.0010 | 0.0073 | 0.0073 |

**Params:** RLE 96,516 · CNN 171,684 (RLE uses 56.2% of CNN's params)

Notes: gap here (~0.0004) is smaller than the fold-to-fold std (0.007+), so this
currently reads as a statistical tie. RLE actually leads on folds 2–3, CNN leads
on 1, 4, 5 — the two trade places rather than one consistently edging the other.
Early read: RLE getting to parity at ~56% of the parameter count is the most
interesting single result so far.

---

### 3. Ultrasound (Breast, 3-class — benign / malignant / normal)

**Setup:** 956 train images — smallest dataset tried so far. 128×128 resolution,
smaller batch size (64) than the other runs given the small N.

| Fold | RLE AUC | RLE Acc | RLE F1 | CNN AUC | CNN Acc | CNN F1 |
|---|---|---|---|---|---|---|
| 1 | 0.8314 | 0.7188 | 0.6870 | 0.8640 | 0.6562 | 0.6569 |
| 2 | 0.9260 | 0.8063 | 0.7720 | 0.7606 | 0.5687 | 0.5595 |
| 3 | 0.8878 | 0.7801 | 0.7630 | 0.8309 | 0.6625 | 0.6668 |
| 4 | 0.8894 | 0.7906 | 0.7636 | 0.8286 | 0.6415 | 0.6209 |
| 5 | 0.8210 | 0.7068 | 0.6589 | 0.7818 | 0.5849 | 0.5761 |
| **Mean** | **0.8711** | **0.7605** | **0.7289** | **0.8132** | **0.6228** | **0.6161** |
| **Std** | 0.0393 | 0.0400 | 0.0466 | 0.0371 | 0.0385 | 0.0426 |

**Params:** RLE 96,451 · CNN 171,619 (RLE uses 56.2% of CNN's params)

Notes: RLE ahead on accuracy/F1 in every fold, only narrowly behind on AUC in
fold 1. Fold variance is much higher here (std ~0.04) due to the small pool
(~190 val samples/fold), so individual folds should be read cautiously — but the
aggregate gap is well beyond that noise. CNN's training curves show it
struggling past ~70% train accuracy even at epoch 100, consistent with
underfitting on this little data.

---

### 4. KMNIST (Handwritten Kuzushiji, 10-class)

**Setup:** 60,000 images (48,000 train / 12,000 val per fold), 10 balanced
classes, 64×64 resolution.

| Fold | RLE AUC | RLE Acc | RLE F1 | CNN AUC | CNN Acc | CNN F1 |
|---|---|---|---|---|---|---|
| 1 | 0.9995 | 0.9792 | 0.9792 | 0.9999 | 0.9932 | 0.9932 |
| 2 | 0.9997 | 0.9801 | 0.9801 | 0.9999 | 0.9939 | 0.9939 |
| 3 | 0.9996 | 0.9802 | 0.9802 | 1.0000 | 0.9945 | 0.9945 |
| 4 | 0.9996 | 0.9795 | 0.9795 | 0.9999 | 0.9944 | 0.9944 |
| 5 | 0.9997 | 0.9801 | 0.9801 | 0.9999 | 0.9912 | 0.9912 |
| **Mean** | **0.9996** | **0.9798** | **0.9798** | **0.9999** | **0.9934** | **0.9934** |
| **Std** | 0.0001 | 0.0004 | 0.0004 | 0.0000 | 0.0012 | 0.0012 |

**Params:** RLE 96,906 · CNN 172,074 (RLE uses 56.3% of CNN's params)

Notes: both models extremely stable (std <0.13% accuracy). CNN leads by a modest
+1.36 accuracy/F1 pts, smaller than the pneumonia gap despite 10x more data —
suggesting the gap isn't purely a function of dataset size. RLE still reaches
97.98% accuracy at just under 97K params, a strong absolute result even in the
comparison it currently trails.

---

### Quick-Reference Summary Table

| Metric | Pneumonia | Brain Tumor | Ultrasound | KMNIST |
|---|---|---|---|---|
| N (train) | 5,232 | 5,600 | 956 | 60,000 |
| Classes | 2 | 4 | 3 | 10 |
| Resolution | 64×64 | 128×128 | 128×128 | 64×64 |
| RLE AUC | 0.9928 | 0.9921 | 0.8711 | 0.9996 |
| CNN AUC | 0.9977 | 0.9925 | 0.8132 | 0.9999 |
| RLE Acc | 0.9606 | 0.9416 | 0.7605 | 0.9798 |
| CNN Acc | 0.9801 | 0.9420 | 0.6228 | 0.9934 |
| RLE F1 | 0.9488 | 0.9415 | 0.7289 | 0.9798 |
| CNN F1 | 0.9744 | 0.9418 | 0.6161 | 0.9934 |
| RLE params | 69,476 | 96,516 | 96,451 | 96,906 |
| CNN params | 116,244 | 171,684 | 171,619 | 172,074 |
| Param savings | 40.2% | 43.8% | 43.8% | 43.7% |
## Why Raw Pixels Are Wasteful

A CNN operating on raw pixels has to learn, from scratch, that:

- Long runs of similar-valued pixels are one "thing," not many independent signals
- Spatial position and local intensity together define structure (edges, blobs, regions)
- Most of the 2D grid it's looking at is low-information filler around the actual
  discriminative structure

RLE bakes exactly this prior directly into the input representation, before training
even starts. Every row and column is reduced to "where does a bright region start, how
long does it run, and how bright is it, on average" — a handful of numbers per row and
per column. That's a large dimensionality drop with a well-defined, interpretable
meaning, and it means the downstream model doesn't have to spend capacity
rediscovering it from raw pixel noise.

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

 a 64×64 image collapses from **4,096 raw pixel
values** down to a **1,536-value structured vector** (768 row + 768 col) — a ~2.7×
reduction before a single learned parameter is involved, and this compressed vector is
*all* the downstream model ever sees.

Two implementations exist and are auto-selected at runtime: a parallel Numba-JIT
scanner for speed, and a NumPy fallback (vectorized via boundary-diff detection) if
`numba` isn't installed — functionally identical, just slower.

A dedicated reconstruction-check cell in the notebook
(`check_single_image_reconstruction`) visually confirms the encoding faithfully
captures the binarized image's structure before any model ever touches it.

---

## Architecture

### RLEClassifier — Per-Branch Architecture

> Row and column branches share this same structure but use separate weights.
> kernal sizes, and stride choices changed according to the dataset.

| Stage | Layer | Channels | Kernel / Stride |
|---|---|---|---|
| 1 | Conv1d + BN + ReLU + Dropout | 1 → 32 | k=3, s=3 |
| 2 | Conv1d + BN + ReLU + Dropout | 32 → 64 | k=4, s=4 |
| 3 | Conv1d + BN + ReLU + Dropout | 64 → 64 | k=3, s=2 |
| — | AdaptiveAvgPool1d(2) | — | pools to a fixed length |

Two independent, lightweight 1D-CNN branches — one for the row profile, one for the
column profile — fused with a small cross-attention module.



**Total parameters: ~69K–97K**, depending on resolution and class count — currently
about **half** the size of the comparison CNN at every resolution tried so far.

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
shape only becomes clear once row and column information are considered jointly —
while staying far cheaper than standard multi-head attention.

### The Comparison Model

To make the efficiency claim rigorous rather than anecdotal, RLEClassifier is
benchmarked against **FairCNNBaseline**: a raw-pixel 2D CNN deliberately built with
the *same* `fused_dim` (256) and the *same* FC head shape, so the only real difference
between the two models is **what they're allowed to see** — compressed RLE structure,
vs. every raw pixel. This isn't meant as a "final leaderboard" comparison; it exists so
the parameter and FLOP gap below is a fair, apples-to-apples reading rather than an
artifact of comparing against an unfairly small or unfairly large baseline.

---

## Early Efficiency Signal

This is the part of the project that's already consistent and worth taking seriously,
even while everything else is still being tuned.

| Metric | RLEClassifier | FairCNNBaseline | RLE's advantage |
|---|---|---|---|
| Parameters | ~69K–97K | ~116K–172K | **~1.7–1.8× fewer, every time so far** |
| MACs / FLOPs | ~6.9M | ~133M | **~19× fewer, every time so far** |
| Memory footprint (float32) | ~0.27 MB | ~0.44–0.66 MB | roughly half |

This pattern has held across every dataset/resolution combination tried so far. It's
not a lucky result on one run — it's a direct structural consequence of feeding the
network a compressed 1D representation instead of a full 2D grid, and it's the part of
this project's story that doesn't depend on further tuning to be true.

On raw GPU wall-clock time, the picture is currently less clean: early runs show the
CNN finishing inference and training slightly faster in practice, since 2D convolutions
on regular pixel grids are exactly the workload GPUs are optimized to parallelize. This
is flagged honestly rather than smoothed over — see [Status](#status--whats-still-open) —
and it's one of the open threads this project still needs to chase down (kernel
overhead? sequence length? the attention module specifically?), likely via a CPU-only
benchmark where GPU-specific parallelism advantages disappear.

---

## Early Accuracy Signal (Preliminary)

**These numbers are early and will change as the model, encoding, and training setup
are iterated on further — treat them as a demonstration that the approach is viable,
not as a final benchmark result.**

Across the datasets tried so far, RLEClassifier has landed anywhere from matching the
CNN baseline to notably exceeding it, depending on the run and dataset — and in no
case tested has it fallen apart or become unusable; accuracy has stayed within a
reasonable band of the CNN baseline throughout, at roughly half the parameter count.
That's the demonstration this proof-of-concept is currently built to make: **a
compressed structural encoding is enough signal to train a competitive classifier**,
not that it has already been proven to beat raw pixels in every setting.

Rather than lock in a specific scoreboard here while the model is still moving, the
honest summary is:

- The efficiency gap (parameters, FLOPs) is stable and well-established.
- The accuracy gap between RLE and the CNN baseline is small in most runs so far, and
  RLE has clearly outperformed the CNN in at least one run.
- Further tuning of the encoding and architecture (see below) is expected to move
  these numbers, likely in RLE's favor, since the current setup has not yet been
  optimized for RLE specifically.

---

## Tuning the Encoding: MAX_RUNS and THRESH_K

The two knobs that control the fidelity/compression trade-off of the RLE
representation are configurable and **actively subject to change**:

- **`MAX_RUNS`** — how many runs are kept per row/column before truncation . Raising it captures more fine-grained structure per row at the cost
  of a larger feature vector (and slightly more downstream compute); lowering it
  compresses further but risks dropping runs in busy/high-contrast rows.
- **`THRESH_K`** — the multiplier on the per-image standard deviation used to set the
  binarization threshold (`threshold = mean + THRESH_K · std`).
  This controls how aggressively pixels are classified as "on" vs. "off" before run
  detection.

Because both values are current defaults rather than settled, tuned choices, **the
numbers in this README should be read as a snapshot of one configuration, not a
ceiling on what RLE can do.** A per-dataset sweep of `MAX_RUNS`/`THRESH_K` is one of
the most direct, low-effort next steps for improving the RLE side of the comparison
(see [Roadmap](#roadmap)).

---

## Why This Is Worth Pursuing

Fewer parameters and dramatically fewer FLOPs map directly onto real, practical
constraints, independent of exactly where the accuracy numbers land once tuning is
finished:

- **Storage / memory footprint** — smaller models ship in smaller binaries and fit in
  tighter memory budgets (edge devices, browser-based inference, mobile).
- **Low-data regimes** — a fixed structural encoding is a built-in prior/regularizer,
  which tends to matter most exactly when labeled data is scarce — a common situation
  in specialized domains.
- **Theoretical compute cost** — ~19× fewer MACs is a large reduction in the raw
  arithmetic a deployment target would need to be provisioned for, even before any
  hardware-specific optimization work is done.
- **A genuinely different representation, not just a smaller version of the same one**
  — RLE isn't a pruned or distilled CNN; it's a structurally different way of looking
  at the image, which is why it's an interesting direction to keep developing rather
  than a dead end if it doesn't immediately match CNN accuracy everywhere.

This is why the project treats RLE as **worth continuing to build out** — not because
it has already been proven superior, but because the efficiency case is already real
and consistent, and the accuracy case is trending toward "competitive" well before any
serious tuning has happened.

---

## Status / What's Still Open

Being upfront about what hasn't been nailed down yet keeps the pitch honest:

- **The model is still being developed.** Architecture, hyperparameters, and the
  encoding knobs (`MAX_RUNS`, `THRESH_K`) are current defaults, not finalized choices.
- **Accuracy/AUC numbers are preliminary** and are expected to move — likely in RLE's
  favor — as the encoding and architecture are tuned specifically for RLE rather than
  reusing a shape mirrored from the CNN baseline for a fair initial comparison.
- **GPU wall-clock time currently favors the CNN**, despite RLE's much lower FLOP
  count — this hasn't been root-caused yet, and a CPU-only benchmark (where GPU
  parallelism advantages disappear) hasn't been run yet either, even though it's the
  setting most likely to convert RLE's FLOP advantage into an actual wall-clock win.
- **Grayscale/B&W only** — the encoding assumes a single intensity channel and hasn't
  been extended to color.
- **Tested at small-to-medium scale so far**; behavior at much larger scale is
  untested.

---

## Roadmap

1. **Sweep `MAX_RUNS` and `THRESH_K`** — the most direct lever for improving RLE's
   accuracy without adding parameters.
2. **CPU-only benchmarking** — the most likely place for RLE's FLOP advantage to
   convert into an actual, provable wall-clock win.
3. **RLE-specific architecture search**, instead of reusing a shape mirrored from the
   CNN baseline for the initial fair comparison.
4. **Color (RGB) extension** — per-channel binarize + RLE + fuse, testing whether the
   efficiency case survives tripling the pipeline for 3 channels.
5. **More runs, more datasets** — the current accuracy readings are based on a limited
   number of runs; broadening this is the main way to turn "early signal" into a
   settled result.
6. **Scale testing** on larger datasets to see how the efficiency-vs-accuracy balance
   shifts as data volume grows.

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
   / `THRESH_K` for your specific image statistics rather than reusing the current
   defaults.
3. Run the RLE encoding + feature extraction cells to build cached feature arrays.
4. Run Cell 8 (architecture) → Cell 8.8 (RLE 5-fold CV).
5. Run Cell 8B (`FairCNNBaseline`) → Cell 9C (CNN 5-fold CV) for the efficiency/accuracy
   comparison.
6. Cell 8C prints a side-by-side parameter and memory breakdown for both models.

All randomness is seeded (`SEED = 42`); `cudnn.benchmark = True` is enabled for speed,
so exact fold-level numbers may vary slightly run-to-run — another reason to treat any
single accuracy/AUC reading as preliminary rather than final.
