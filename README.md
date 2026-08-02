# NetGen — Batch Neural Network Model Generator & Manager

Generate diverse PyTorch model training folders with verified parameter counts,
then manage, compare, and benchmark them.

Use cases: benchmarking, NAS prototyping, education, training pipeline testing.

## Quick Start

```bash
pip install -e .
```

## Commands

```
netgen generate --range <min>-<max> --count <N> [options]
netgen list     [--dir <path>]
netgen info     <id> [--dir <path>]
netgen compare  [--dir <path>] [--sort params|accuracy|loss] [--top N]
netgen train    <id> [--epochs N] [--lr X] [--device cuda,mps]
netgen eval     <id>
netgen benchmark [--dir <path>] [--epochs N] [--lr X] [--device cuda,mps]
netgen clean    [--dir <path>] [--untrained] [--dry-run|--force]
netgen export   [--dir <path>] [--format md|csv|json]
netgen archs    [--tree|--list]
```

### `generate` — Create model folders

| Option | Default | Description |
|--------|---------|-------------|
| `--range` | *required* | Parameter count range, e.g. `10000-20000` |
| `--count` | 20 | Number of models |
| `--output` `-o` | `./generated_models` | Output directory |
| `--arch` | all | Filter architectures, e.g. `mlp,cnn,lstm` |
| `--preset` `-p` | — | Quick filter: `cv`, `nlp`, `gen`, `light` |
| `--dataset` | `syn` | Dataset name |
| `--seed` | 42 | Random seed |
| `--jobs` `-j` | 1 | Parallel workers |
| `--device` | cuda,mps | Training device priority, e.g. `cuda,mps`. `cpu` is always the final fallback (may be omitted: `cuda,cpu` == `cuda`) |

```bash
netgen generate --range 10000-20000 --count 20
netgen generate --device cuda,mps --range 10000-20000 --count 20
netgen generate --device cpu --range 10000-20000 --count 20   # CPU only
netgen generate --preset cv --range 5000-50000 --count 10
netgen generate --preset nlp --arch lstm --range 10000-50000 --count 5
netgen generate --range 5000000000-10000000000 --count 3
netgen generate --range 1K-50K --count 5 --dataset iris     # real features
netgen generate --range 1K-50K --count 3 --dataset mnist    # 2-D CNNs & flat nets
netgen generate --range 1K-50K --count 3 --dataset cifar10  # adds vit/mixer
```

`--dataset` picks the data source (`syn` by default). Real datasets:
`iris`, `wine`, `breast_cancer`, `moons`, `circles`, `blobs`, `mnist`,
`cifar10`, `text`, `line`. The dataset fixes the model's input dimensions
and incompatible architectures are filtered automatically:

- **Vector datasets** (iris, wine, ...): supervised vector nets (`mlp`,
  `linear`, `wide`, `deep`, `resblock`, `moe`, `unary`) — inputs rewritten
  to the dataset's feature count.
- **Image datasets** (mnist, cifar10): 2-D nets (`cnn`, `rescnn`, `sepcnn`,
  `densecnn`, `unet`) keep the image shape (`Conv2d(1|3, ...)`); everything
  else gets **flattened samples** (MNIST → 784-D vectors). `cifar10` also
  allows the RGB patch nets `vit`/`mixer`. The first training run downloads
  torchvision data (one time).
- Image-dataset `eval.py` measures **held-out test-set accuracy**
  (`SynData(train=False)`).
- Architectures with special data contracts (RNN sine sequences, gpt/t5
  tokens, GCN graphs, GAN/self-supervised/multitask) keep their synthetic
  data and are excluded from real-dataset runs.

### `benchmark` — Train all & rank

```bash
netgen benchmark --epochs 10
netgen benchmark --epochs 20 --lr 0.01 --batch-size 128
netgen benchmark --device cuda,mps   # override device for all models
netgen benchmark --workers 4        # train 4 models concurrently
netgen benchmark --force            # re-train already-trained models
netgen benchmark --time-budget 30   # ~30min wall clock, split across models
```

- Each model is trained with a **20% validation split** (`VAL_SPLIT` in
  `config.py`); `training_log.md` gains `Val Loss` / `Val Acc` columns and
  ranking uses **validation metrics** (generalization, not train-set fit).
- Failed models are retried once automatically (`--retries N`).
- `--time-budget MIN` splits the budget across models as per-model timeouts;
  models that exceed it retry once with half the epochs.
- Saves `benchmark_report.md` + a `benchmark_curves.png` loss-curve chart.

### `sweep` — Hyperparameter search for one model

```bash
netgen sweep 001                    # default grid: lr × batch
netgen sweep 001 --epochs 10 --lrs 0.001,0.01,0.0001 --batches 32,64,128
```

Tries every lr × batch_size combo with the model's own `train.py`, ranks by
validation metric, **re-trains the winner** (final `model.pth`) and writes
the best hyperparameters back into `config.py` — so plain
`netgen train 001` afterwards uses them. Results are saved to
`sweep_report.md`.

### `train` / `eval` — Train or evaluate a single model

```bash
netgen train 001                    # train with the model's config defaults
netgen train 001 --epochs 50 --lr 0.001 --device cpu
netgen eval 001                     # run the model's eval.py
```

`--device` overrides the device priority baked in at generation time.

### `list` / `info` / `compare` / `clean` / `export` — Manage models

```bash
netgen list                          # Table of all models + training status
netgen info 001                      # Detailed view of one model
netgen compare --sort accuracy       # Rank trained models
netgen clean --untrained --dry-run   # Preview removal
netgen clean --keep-best 3 --force   # Keep only top 3
netgen export --format md            # Save comparison report
```

### `archs` — Browse architectures

```bash
netgen archs          # Family tree (9 families, 31 architectures)
netgen archs --list   # Flat list with descriptions
```

## File Architecture (3 Tiers)

| Tier | Params | Files | Highlights |
|------|--------|:-----:|-----------|
| **Quick** | < 50K | 8 | Basic training, `training_log.md`, `data_explore.py`, checkpoints & best model |
| **Standard** | 50K ~ 50M | 12 | + lr scheduler, early stopping, sweep, real visualize |
| **Production** | > 50M | 17+ | + DDP, AMP, model sub-package, benchmark, profile, ONNX export |

### Quick Tier

```
001-mlp-200x150x80/
├── config.py         # All hyperparameters (LR, epochs, optimizer, scheduler...)
├── data.py           # Dataset loader
├── data_explore.py   # One-click data inspection
├── model.py          # PyTorch nn.Module
├── train.py          # Training (all CLI args supported)
├── eval.py           # Evaluation
├── predict.py        # Inference demo
├── visualize.py      # Reads training_log.md, plots real curves
├── checkpoints/      # Periodic checkpoints
├── best_model.pth    # Auto-saved (lowest loss)
├── model.pth         # Final model
├── requirements.txt
└── README.md         # Task-specific summary
```

### Standard Tier (adds)

```
├── sweep.py          # Grid search lr × batch_size
└── (all Quick-tier features included)
```

### Production Tier (adds)

```
├── model/            # Modular sub-package
├── configs/          # YAML presets
├── scripts/          # benchmark, profile, ONNX export
├── logs/             # Detailed CSV logs
└── (all Standard-tier features included)
```

## Running a Model

```bash
cd 001-mlp-200x150x80

# 1. Inspect the data
python data_explore.py

# 2. Train
python train.py --epochs 50 --lr 0.001 --batch-size 128

# 3. Resume from checkpoint
python train.py --resume checkpoints/ckpt_0010.pth --epochs 100

# 4. Evaluate
python eval.py

# 5. Visualize
python visualize.py
```

### Training CLI Reference

```
python train.py [options]

--lr FLOAT            Learning rate (default: 0.001)
--epochs INT          Training epochs (default: 30)
--batch-size INT      Batch size (default: 64)
--save-every INT      Checkpoint interval (default: 10)
--optimizer STR       adam | sgd | adamw
--weight-decay FLOAT  L2 regularization
--momentum FLOAT      Momentum for SGD
--scheduler STR       none | cosine | plateau | step
--patience INT        Early stopping patience
--grad-clip FLOAT     Gradient clipping max norm
--seed INT            Random seed
--resume PATH         Resume from checkpoint
```

Architecture-specific config: VAE has `BETA`, GAN has `G_LR`/`D_LR`, Contrastive has `TEMPERATURE`, Siamese has `MARGIN`, Classification has `LABEL_SMOOTHING`.

## Dataset Compatibility

`--dataset` auto-filters architectures and rewrites model dimensions:

| Dataset | Auto-selected Architectures |
|---------|----------------------------|
| iris, wine, breast_cancer, moons, circles, blobs | `linear` `mlp` `wide` `deep` `resblock` `moe` `unary` |
| line | same as above (regression) |
| mnist | 2-D: `cnn` `rescnn` `sepcnn` `densecnn` `unet` + flat: same as above |
| cifar10 | mnist set + `vit` `mixer` |
| text | `gpt` `t5` |
| syn | all 31 architectures |

## Architectures (31 total)

### Universal (any range) — 8

`mlp` `deep` `wide` `resblock` `highway` `moe` `transformer` `sae`

### Classic (up to 5M–10M) — 12

`unary` (a/b/c) `linear` `cnn` `lstm` `gru` `bilstm` `ae` `vae` `gan` `multitask` `contrastive` `siamese`

### Medium (≥ 100K) — 6

`rescnn` `sepcnn` `densecnn` `attnlstm` `selfattn` `gcn`

### Large (≥ 10M) — 5

`vit` `unet` `mixer` `gpt` `t5`

## Datasets

| Name | Description | Features | Classes |
|------|-------------|----------|---------|
| `syn` | Synthetic Gaussian | configurable | configurable |
| `iris` | Iris flowers | 4 | 3 |
| `wine` | Wine cultivars | 13 | 3 |
| `breast_cancer` | Breast cancer diagnosis | 30 | 2 |
| `moons` | Two half-circles | 2 | 2 |
| `circles` | Concentric circles | 2 | 2 |
| `blobs` | Gaussian blobs | 6 | 5 |
| `mnist` | Handwritten digits | 1×28×28 | 10 |
| `cifar10` | Natural images | 3×32×32 | 10 |
| `text` | Character sequences | 20 | 10 |
| `line` | Linear regression | 1 | 1 |

## Python API

```python
from netgen import find_candidates, gen_folder, list_architectures

candidates = find_candidates(lo=10000, hi=20000, count=10, seed=42,
                             arch_filter=['mlp', 'cnn', 'lstm'])

for desc, code, params, inp, outp, mtype in candidates:
    gen_folder('./output', 1, desc, code, 'M{}', params, inp, outp, mtype,
               dataset='iris', device_priority=['cuda', 'mps'])
```

`device_priority` (list) is written into the generated `config.py` as
`DEVICE_PRIORITY`; each script resolves `DEVICE` at import time by taking the
first available device in that order, with `cpu` always as the final fallback.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- numpy
- matplotlib (for visualize.py)
- scikit-learn (optional)
- torchvision (optional)
