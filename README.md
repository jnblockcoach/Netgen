# NetGen — Batch Neural Network Model Generator

Generate diverse PyTorch model training folders with verified parameter counts.

Use cases: benchmarking, NAS prototyping, education, training pipeline testing.

## Quick Start

```bash
pip install -e .
```

## Usage

```
python -m netgen --range <min>-<max> --count <N> [--output <dir>] [--arch <filter>] [--dataset <name>] [--seed <S>]
```

| Option | Required | Default | Description |
|--------|:-------:|---------|-------------|
| `--range` | ✓ | — | Parameter count range, e.g. `10000-20000` |
| `--count` | | 20 | Number of models to generate |
| `--output` `-o` | | `./generated_models` | Output directory |
| `--arch` | | all | Comma-separated filter, e.g. `mlp,cnn,lstm` |
| `--dataset` | | `syn` | Dataset: `syn`, `iris`, `wine`, `breast_cancer`, `moons`, `circles`, `blobs`, `mnist`, `cifar10`, `text`, `line` |
| `--seed` | | 42 | Random seed |

### Examples

```bash
# 20 models with 10K–20K params
python -m netgen --range 10000-20000 --count 20

# Only MLP and CNN
python -m netgen --range 5000-50000 --count 10 --arch mlp,cnn

# With real dataset
python -m netgen --range 1000-10000 --count 5 --dataset iris

# Huge models
python -m netgen --range 5000000000-10000000000 --count 3
```

## File Architecture (3 Tiers)

Generated folders scale with parameter count:

| Tier | Params | Files | Capabilities |
|------|--------|:-----:|-------------|
| **Quick** | < 50K | 8 | Basic training loop, `training_log.md`, `data_explore.py` |
| **Standard** | 50K ~ 50M | 12 | + lr scheduler, early stopping, checkpoints, sweep, visualize |
| **Production** | > 50M | 17+ | + DDP, AMP, model sub-package, benchmark, profile, ONNX export |

### Quick Tier

```
001-mlp-200x150x80/
├── config.py         # Hyperparameters (LR, epochs, dims, dataset)
├── data.py           # Dataset loader
├── data_explore.py   # Inspect the dataset — stats & shapes
├── model.py          # PyTorch nn.Module
├── train.py          # Training (argparse: --lr --epochs --batch-size)
├── eval.py           # Evaluation
├── predict.py        # Inference demo
├── visualize.py      # Training curve plot
├── requirements.txt
└── README.md         # Task-specific summary
```

### Standard Tier (adds)

```
├── sweep.py          # Grid search lr × batch_size
├── visualize.py      # Reads training_log.md (real data)
├── predict.py        # Batch inference from best_model.pth
├── checkpoints/      # Periodic checkpoints
└── best_model.pth    # Early-stopping best model
```

### Production Tier (adds)

```
├── model/            # Modular sub-package
│   ├── __init__.py
│   └── layers.py
├── configs/          # YAML presets
│   ├── default.yaml
│   └── large.yaml
├── scripts/
│   ├── benchmark.py  # Latency & throughput
│   ├── profile.py    # FLOPs & memory
│   └── export.py     # ONNX / TorchScript
├── logs/             # Detailed CSV logs
└── checkpoints/
```

## Running a Model

```bash
cd 001-mlp-200x150x80

# 1. Inspect the data
python data_explore.py

# 2. Train (auto-generates training_log.md)
python train.py --epochs 50 --lr 0.001 --batch-size 128

# 3. Evaluate
python eval.py

# 4. Visualize (Standard+) — reads training_log.md
python visualize.py

# 5. Hyperparameter sweep (Standard+)
python sweep.py
```

## Dataset Compatibility

`--dataset` auto-filters architectures and rewrites model dimensions:

| Task | Datasets | Auto-selected Architectures |
|------|----------|---------------------------|
| Classification | iris, wine, breast_cancer, moons, circles, blobs, mnist, cifar10, text | linear, mlp, deep, wide, resblock, highway, moe, cnn, multitask |
| Regression | line | linear, mlp, deep, wide, resblock |
| Synthetic | syn | all 31 architectures |

## Architectures (31 total)

### Universal (any param range)

| Key | Architecture | Scaling |
|-----|-------------|---------|
| `mlp` | 3-layer MLP | d1·d2 + d2·d3 |
| `deep` | Deep MLP | N × d² |
| `wide` | Wide Net | d_in × d_hidden |
| `resblock` | Residual Blocks | N × 2·d·h |
| `highway` | Highway Network | N × 2·d² |
| `moe` | Mixture of Experts | E × 2·d·h |
| `transformer` | Transformer Encoder | L × d_model² |
| `sae` | Stacked Autoencoder | N × 2·h² |

### Classic (up to 5M–10M params)

| Key | Architecture | Max Params |
|-----|-------------|:----------:|
| `unary` | 1-param variants (a/b/c) | 1 |
| `linear` | Single linear layer | 5M |
| `cnn` | Convolutional (8×8) | 5M |
| `lstm` | LSTM | 5M |
| `gru` | GRU | 5M |
| `bilstm` | BiLSTM | 5M |
| `ae` | Autoencoder | 5M |
| `vae` | Variational AE | 5M |
| `gan` | GAN | 10M |
| `multitask` | Multi-Task | 10M |
| `contrastive` | Contrastive Learning | 10M |
| `siamese` | Siamese Network | 10M |

### Medium (≥ 100K params)

| Key | Architecture | Description |
|-----|-------------|-------------|
| `rescnn` | Residual CNN | Multi-stage with skip connections |
| `sepcnn` | Separable CNN | Depthwise + pointwise (MobileNet style) |
| `densecnn` | Dense CNN | Dense connections (DenseNet style) |
| `attnlstm` | Attention LSTM | LSTM + multi-head self-attention pooling |
| `selfattn` | Self-Attention | Pure attention, no RNN/CNN |
| `gcn` | Graph ConvNet | 2-layer GCN for node classification |

### Large (≥ 10M params)

| Key | Architecture | Description |
|-----|-------------|-------------|
| `vit` | Vision Transformer | Patch embedding + Transformer encoder |
| `unet` | U-Net | Encoder-decoder with skip connections |
| `mixer` | MLP-Mixer | Token-mixing + channel-mixing MLPs |
| `gpt` | GPT Decoder | Causal transformer for language modeling |
| `t5` | T5 Encoder-Decoder | Full encoder-decoder transformer |

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

# Search architectures
candidates = find_candidates(lo=10000, hi=20000, count=10, seed=42,
                             arch_filter=['mlp', 'cnn', 'lstm'])

# Generate a folder
for desc, code, params, inp, outp, mtype in candidates:
    gen_folder('./output', 1, desc, code, 'M{}', params, inp, outp, mtype,
               dataset='iris')
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- numpy
- matplotlib (for visualize.py)
- scikit-learn (optional, for real datasets)
- torchvision (optional, for MNIST / CIFAR-10)
