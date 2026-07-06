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

- [What Is Run-Length Encoding (RLE) Here?](#what-is-run-length-encoding-rle-here)
- [Why RLE Features Can Compete with CNN Features](#why-rle-features-can-compete-with-cnn-features)
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

## What Is Run-Length Encoding (RLE) Here?

In this project, each row and each column of an image is broken into **runs** — 
continuous stretches of bright ("on") pixels — and every run is reduced to just
three numbers:

| Feature | What it is |
|---|---|
| `start` | Where the run begins along the row/column (÷ image width) |
| `length` | How long the run is (÷ image width) |
| `mean_intensity` | Average pixel value across the run (÷ 255) |

Up to `MAX_RUNS` runs are kept per row/column (padded with zeros if there are
fewer, truncated if there are more), so every row/column always contributes a
fixed `MAX_RUNS × 3` numbers regardless of how much or how little structure it
contains.

This is done once for the rows and once for the columns (columns via a transposed
scan), and the two resulting sets of numbers are flattened and concatenated into a
single 1D vector:

```
row_vector = MAX_RUNS × 3 numbers per row     (all rows flattened)
col_vector = MAX_RUNS × 3 numbers per column  (all columns flattened)
feature_vector = concat([row_vector, col_vector])
```

For a 64×64 image with `MAX_RUNS = 4`: 768 values from rows + 768 from columns =
**a 1,536-number 1D vector**, replacing the original 4,096-pixel 2D grid. That
vector — `start`, `length`, and `mean_intensity` per run, per row and per column —
is the entire input `RLEClassifier` ever sees.

---

## Why RLE Features Can Compete with CNN Features

**Position is explicit, not implicit.** A standard CNN filter is translation-equivariant
— the same weights slide over every location in the image, so the filter itself has no
built-in notion of "this is near the top" vs. "this is near the bottom." Any sense of
position has to emerge indirectly, through boundary/padding effects and how features
get combined across layers — it's never handed to the network as a direct value. In
the RLE feature set, position **is** a direct value: `start` records exactly where
each run begins along its row/column. Position is a first-class number from the very
first feature, rather than something the network has to reconstruct indirectly.

**Intensity is summarized directly.** Each run's `mean_intensity` gives the model a
direct per-region average brightness, computed once, up front — rather than raw pixel
values the CNN has to aggregate through multiple conv/pooling layers before it has any
regional intensity summary at all.

**Start + length together sketch shape.** Where a run begins (`start`) and how far it
extends (`length`), read across many rows and columns, trace the rough outline/extent
of a region — where it starts, how wide or tall it is — without needing learned edge
filters to detect that structure first.

**A note on "bright" vs. "dark" regions.** The current encoding looks for runs of "on"
pixels after thresholding (`binary = image > threshold`), which by default picks out
*brighter* regions. Whether that's the diagnostically useful direction depends on the
dataset — an X-ray opacity tends to show up brighter, while a malignant ultrasound
region is often *darker* (hypoechoic) than surrounding tissue. This means the threshold
direction is something to check and potentially flip per dataset, not something to
assume is always "bright = signal."

**What this means for competing with CNNs.** Because the RLE feature set already
encodes position, regional intensity, and rough shape/extent directly, a network built
on it doesn't have to spend capacity learning to extract those things from raw pixels
the way a CNN does. This has already paid off outright on the smallest dataset tested:
on Ultrasound (956 images), RLE beat the CNN baseline by **+13.8 accuracy points** —
not a tie, not "close," a clear win, with RLE ahead on accuracy and F1 in every single
fold. That result stands on its own.

The broader expectation this project is built on is that this isn't a one-off limited
to small datasets: with the right architecture and per-dataset tuning (encoding
direction, `MAX_RUNS`, `THRESH_K`), RLE should be able to match or exceed CNN accuracy
more consistently — since it starts with a structural head start (position, intensity,
shape) rather than raw pixels. That broader, across-the-board claim is still being
tested: on the other three datasets tried so far, RLE currently ties (Brain Tumor) or
trails by 1–2 points (Pneumonia, KMNIST) using an architecture that was deliberately
mirrored to the CNN's shape for a fair first comparison, not yet tuned specifically for
RLE. The Ultrasound win is the concrete evidence this broader expectation is built on —
the next step is testing whether RLE-specific tuning closes the remaining gap
elsewhere too.

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

At the current default `MAX_RUNS = 4`, a 64×64 image collapses from **4,096 raw pixel
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

- **`MAX_RUNS`** — how many runs are kept per row/column before truncation (current
  default: `4`). Raising it captures more fine-grained structure per row at the cost
  of a larger feature vector (and slightly more downstream compute); lowering it
  compresses further but risks dropping runs in busy/high-contrast rows.
- **`THRESH_K`** — the multiplier on the per-image standard deviation used to set the
  binarization threshold (`threshold = mean + THRESH_K · std`, current default: `0.2`).
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
