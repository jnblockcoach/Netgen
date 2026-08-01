"""File generation: write model.py, config, train, eval, predict, visualize, etc.

Each model type gets customized files — different architectures have different
training objectives, metrics, and data handling.
"""
import os
import re
from typing import Optional, Dict

from .templates import get_templates, get_extra_files
from .datasets import get_dataset_code


# ── Tier determination ──

def _get_tier(params: int) -> str:
    """Determine file architecture tier based on parameter count."""
    if params < 50_000:
        return 'quick'
    elif params < 50_000_000:
        return 'standard'
    else:
        return 'production'

# ── Architecture-specific metadata ──

_ARCH_META: Dict[str, dict] = {
    'ce': {
        'task': 'Classification',
        'loss': 'CrossEntropyLoss',
        'metric': 'Accuracy',
        'input_desc': 'Feature vectors',
        'output_desc': 'Class labels (0 ~ OUTPUT_DIM-1)',
        'usage': 'python train.py  # then python eval.py  (reports Accuracy)',
    },
    'mse': {
        'task': 'Regression',
        'loss': 'MSELoss',
        'metric': 'MSE (mean squared error)',
        'input_desc': 'Feature vectors',
        'output_desc': 'Continuous target values',
        'usage': 'python train.py  # then python eval.py  (reports MSE)',
    },
    'cnn': {
        'task': 'Image Classification',
        'loss': 'CrossEntropyLoss',
        'metric': 'Accuracy',
        'input_desc': 'Images (INPUT_DIM channels, 8×8)',
        'output_desc': 'Class labels (0 ~ OUTPUT_DIM-1)',
        'usage': 'python train.py  # then python eval.py  (reports Accuracy)',
    },
    'rnn': {
        'task': 'Sequence Classification',
        'loss': 'CrossEntropyLoss',
        'metric': 'Accuracy',
        'input_desc': 'Time series (seq_len × features)',
        'output_desc': 'Class labels (0 ~ OUTPUT_DIM-1)',
        'usage': 'python train.py  # then python eval.py  (reports Accuracy)',
    },
    'ae': {
        'task': 'Autoencoder Reconstruction',
        'loss': 'MSELoss (input vs reconstruction)',
        'metric': 'Reconstruction MSE',
        'input_desc': 'Feature vectors (target = input itself)',
        'output_desc': 'Reconstructed input (same dim as input)',
        'usage': 'python train.py  # then python eval.py  (reports Reconstruction MSE)',
    },
    'vae': {
        'task': 'Variational Autoencoder',
        'loss': 'Reconstruction MSE + KL divergence',
        'metric': 'Reconstruction MSE',
        'input_desc': 'Feature vectors (target = input itself)',
        'output_desc': 'Reconstructed input + latent μ/logvar',
        'usage': 'python train.py  # then python eval.py  (reports Reconstruction MSE)',
    },
    'gan': {
        'task': 'Generative Adversarial Network',
        'loss': 'BCELoss (generator + discriminator)',
        'metric': 'Generator loss / Discriminator loss',
        'input_desc': 'Random noise z ~ N(0,1)',
        'output_desc': 'Generated samples matching real data distribution',
        'usage': 'python train.py  # monitors D-loss and G-loss',
    },
    'mt': {
        'task': 'Multi-Task Classification',
        'loss': 'CrossEntropyLoss (sum over tasks)',
        'metric': 'Accuracy per task',
        'input_desc': 'Feature vectors shared across tasks',
        'output_desc': 'Two class labels (one per task head)',
        'usage': 'python train.py  # then python eval.py  (reports per-task Accuracy)',
    },
    'contrastive': {
        'task': 'Contrastive Learning',
        'loss': 'NT-Xent (contrastive)',
        'metric': 'Contrastive loss',
        'input_desc': 'Feature vectors (pairs are augmented copies)',
        'output_desc': 'Normalized embedding vectors',
        'usage': 'python train.py  # then python eval.py  (reports embedding shape)',
    },
    'siamese': {
        'task': 'Similarity Learning (Siamese)',
        'loss': 'Contrastive (pairwise distance)',
        'metric': 'Mean pairwise distance',
        'input_desc': 'Pairs of feature vectors (x1, x2)',
        'output_desc': 'Distance between paired embeddings',
        'usage': 'python train.py  # then python eval.py  (reports mean distance)',
    },
}

_ARCH_META['deep'] = _ARCH_META['ce']
_ARCH_META['wide'] = _ARCH_META['ce']
_ARCH_META['resblock'] = _ARCH_META['ce']
_ARCH_META['highway'] = _ARCH_META['ce']
_ARCH_META['moe'] = _ARCH_META['ce']
_ARCH_META['transformer'] = _ARCH_META['ce']
_ARCH_META['bilstm'] = _ARCH_META['rnn']
_ARCH_META['gru'] = _ARCH_META['rnn']
_ARCH_META['lstm'] = _ARCH_META['rnn']
_ARCH_META['linear'] = _ARCH_META['mse']
_ARCH_META['unary'] = _ARCH_META['mse']
_ARCH_META['sae'] = _ARCH_META['ae']
# Medium-tier
_ARCH_META['rescnn'] = _ARCH_META['cnn']
_ARCH_META['sepcnn'] = _ARCH_META['cnn']
_ARCH_META['densecnn'] = _ARCH_META['cnn']
_ARCH_META['attnlstm'] = _ARCH_META['rnn']
_ARCH_META['selfattn'] = _ARCH_META['ce']
_ARCH_META['gcn'] = {'task': 'Graph Node Classification', 'loss': 'CrossEntropyLoss',
                     'metric': 'Accuracy', 'input_desc': 'Node feature vectors',
                     'output_desc': 'Node class labels',
                     'usage': 'python train.py  # GCN with adjacency matrix'}
# Large-tier
_ARCH_META['vit'] = {'task': 'Image Classification (Vision Transformer)', 'loss': 'CrossEntropyLoss',
                     'metric': 'Accuracy', 'input_desc': 'RGB images (3×32×32)',
                     'output_desc': 'Class labels',
                     'usage': 'python train.py (supports DDP/AMP)'}
_ARCH_META['unet'] = {'task': 'Image Segmentation', 'loss': 'CrossEntropyLoss',
                      'metric': 'Accuracy', 'input_desc': 'Images (INPUT_DIM×H×W)',
                      'output_desc': 'Pixel-wise class predictions',
                      'usage': 'python train.py (supports DDP/AMP)'}
_ARCH_META['mixer'] = {'task': 'Image Classification (MLP-Mixer)', 'loss': 'CrossEntropyLoss',
                       'metric': 'Accuracy', 'input_desc': 'RGB images (3×32×32)',
                       'output_desc': 'Class labels',
                       'usage': 'python train.py (supports DDP/AMP)'}
_ARCH_META['gpt'] = {'task': 'Language Modeling', 'loss': 'CrossEntropyLoss',
                     'metric': 'Perplexity', 'input_desc': 'Token indices',
                     'output_desc': 'Next-token logits',
                     'usage': 'python train.py (supports DDP/AMP)'}
_ARCH_META['t5'] = {'task': 'Sequence-to-Sequence', 'loss': 'CrossEntropyLoss',
                    'metric': 'Accuracy', 'input_desc': 'Source token indices',
                    'output_desc': 'Target token logits',
                    'usage': 'python train.py (supports DDP/AMP)'}

# ── Constants ──

_ARGPARSE_TRAIN = """
import argparse
from config import *
_args = argparse.ArgumentParser()
_args.add_argument('--lr', type=float, help='Learning rate')
_args.add_argument('--epochs', type=int, help='Number of training epochs')
_args.add_argument('--batch-size', type=int, dest='batch_size', help='Batch size')
_args.add_argument('--save-every', type=int, dest='save_every', help='Checkpoint interval (epochs)')
_args.add_argument('--optimizer', choices=['adam','sgd','adamw'], help='Optimizer')
_args.add_argument('--weight-decay', type=float, dest='weight_decay', help='L2 regularization')
_args.add_argument('--momentum', type=float, help='Momentum (for SGD)')
_args.add_argument('--scheduler', choices=['none','cosine','plateau','step'], help='LR scheduler')
_args.add_argument('--patience', type=int, help='Early stopping patience')
_args.add_argument('--grad-clip', type=float, dest='grad_clip', help='Gradient clipping max norm')
_args.add_argument('--seed', type=int, help='Random seed')
_args.add_argument('--device', type=str, default=None, help='Device priority override, e.g. cuda,mps (cpu always final fallback)')
_args.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
_a = _args.parse_args()
if _a.lr is not None: LR = _a.lr
if _a.epochs is not None: EPOCHS = _a.epochs
if _a.batch_size is not None: BATCH_SIZE = _a.batch_size
if _a.save_every is not None: SAVE_EVERY = _a.save_every
if _a.optimizer is not None: OPTIMIZER = _a.optimizer
if _a.weight_decay is not None: WEIGHT_DECAY = _a.weight_decay
if _a.momentum is not None: MOMENTUM = _a.momentum
if _a.scheduler is not None: SCHEDULER = _a.scheduler
if _a.patience is not None: PATIENCE = _a.patience
if _a.grad_clip is not None: GRAD_CLIP = _a.grad_clip
if _a.seed is not None: SEED = _a.seed
if _a.device is not None:
    DEVICE_PRIORITY = [d.strip().lower() for d in _a.device.split(',') if d.strip()]
    DEVICE = resolve_device()
"""

# ── File I/O ──

def write_file(path: str, content: str) -> None:
    """Write content to file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── Dimension rewriting (for real dataset adaptation) ──

def _rewrite_model_dims(code: str, old_in: int, old_out: int,
                        new_in: int, new_out: int, model_type: str) -> str:
    """Rewrite a model code string to use new input/output dimensions."""
    import re

    if old_in == new_in and old_out == new_out:
        return code

    # Replace the FIRST nn.Linear/CONV2D/LSTM/GRU(old_in, ...) → new_in
    pat_first = rf'(nn\.(?:Linear|Conv2d|LSTM|GRU)\()(\s*){old_in}(\s*,)'
    code = re.sub(pat_first, rf'\g<1>{new_in},', code, count=1)

    # Replace the LAST nn.Linear(..., old_out) → new_out
    pat_last = rf'nn\.Linear\((\s*)\d+(\s*),(\s*){old_out}(\s*)\)'
    matches = list(re.finditer(pat_last, code))
    if matches:
        m = matches[-1]
        old_call = m.group(0)
        new_call = re.sub(rf'\b{old_out}\b', str(new_out), old_call)
        code = code[:m.start()] + new_call + code[m.end():]

    return code


# ── config.py generator ──

def gen_config(input_dim: int, output_dim: int,
               loss_type: str = "ce", dataset: str = "syn",
               model_type: str = "ce",
               device_priority: Optional[list] = None) -> str:
    """Generate config.py with clear comments and architecture-specific params.

    device_priority: training device preference order, e.g. ['cuda', 'mps'].
        'cpu' is ALWAYS the final fallback — it need not be listed.
    """
    dataset_info = _get_dataset_info(dataset)
    base = (
        f"# ===========================================\n"
        f"#  CONFIG — all hyperparameters live here\n"
        f"# ===========================================\n\n"
        f"# ---------- Dataset ----------\n"
        f"DATASET = '{dataset}'        # {dataset_info}\n"
        f"INPUT_DIM = {input_dim}      # number of input features\n"
        f"OUTPUT_DIM = {output_dim}    # number of output classes / target dim\n\n"
        f"# ---------- Training ----------\n"
        f"LR = 0.001                   # learning rate\n"
        f"EPOCHS = 30                  # training epochs\n"
        f"BATCH_SIZE = 64              # batch size\n"
        f"SAVE_EVERY = 10              # checkpoint interval (epochs)\n"
        f"OPTIMIZER = 'adam'           # adam | sgd | adamw\n"
        f"WEIGHT_DECAY = 0.0           # L2 regularization\n"
        f"MOMENTUM = 0.9               # momentum (for SGD)\n"
        f"SCHEDULER = 'none'           # none | cosine | plateau | step\n"
        f"PATIENCE = 10                # early stopping patience\n"
        f"GRAD_CLIP = 1.0              # gradient clipping max norm\n"
        f"SEED = 42                    # random seed\n"
        f"VAL_SPLIT = 0.2              # validation split ratio (0 = no validation)\n"
        f"LOSS_TYPE = '{loss_type}'    # loss function variant\n"
    )

    # Architecture-specific params
    extra = ""
    if model_type in ('vae',):
        extra = "BETA = 0.01                  # KL divergence weight (VAE only)\n"
    elif model_type == 'gan':
        extra = (
            "G_LR = 0.001                # generator learning rate (GAN only)\n"
            "D_LR = 0.001                # discriminator learning rate (GAN only)\n"
        )
    elif model_type == 'contrastive':
        extra = "TEMPERATURE = 0.5           # contrastive temperature\n"
    elif model_type == 'siamese':
        extra = "MARGIN = 1.0                # siamese margin\n"
    elif model_type in ('ce', 'cnn', 'rnn', 'mt'):
        extra = "LABEL_SMOOTHING = 0.0       # label smoothing (classification)\n"

    return base + extra + _gen_device_block(device_priority)


def _gen_device_block(device_priority: Optional[list]) -> str:
    """Generate the DEVICE_PRIORITY + DEVICE resolution block for config.py.

    DEVICE_PRIORITY lists devices in preference order; 'cpu' is always
    appended as the final fallback. DEVICE is resolved at import time so
    every script (train/eval/predict/sweep) picks the same device.
    """
    priority = device_priority or ['cuda', 'mps']
    if 'cpu' not in priority:
        priority = priority + ['cpu']
    return (
        f"\n"
        f"# ---------- Device ----------\n"
        f"# Training device priority (order = preference).\n"
        f"# 'cpu' is ALWAYS the final fallback, so it may be omitted from the list.\n"
        f"# e.g. ['cuda', 'mps'] -> use CUDA if available, else MPS, else CPU.\n"
        f"DEVICE_PRIORITY = {list(priority)}\n"
        f"\n"
        f"import torch\n"
        f"def resolve_device(priority=None):\n"
        f"    '''Return the first available torch.device in the priority list.\n"
        f"    priority overrides DEVICE_PRIORITY when given; 'cpu' always works.\n"
        f"    Used by train.py --device override as well.\n"
        f"    '''\n"
        f"    for _d in (priority if priority is not None else DEVICE_PRIORITY):\n"
        f"        if _d == 'cuda' and torch.cuda.is_available():\n"
        f"            return torch.device('cuda')\n"
        f"        if _d == 'mps' and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():\n"
        f"            return torch.device('mps')\n"
        f"        if _d == 'cpu':\n"
        f"            return torch.device('cpu')\n"
        f"    return torch.device('cpu')\n"
        f"DEVICE = resolve_device()\n"
    )


def _get_dataset_info(dataset: str) -> str:
    infos = {
        'syn': 'synthetic Gaussian random data',
        'iris': 'Iris flower dataset (4 features, 3 classes)',
        'wine': 'Wine cultivar dataset (13 features, 3 classes)',
        'breast_cancer': 'Breast cancer Wisconsin dataset (30 features, 2 classes)',
        'moons': 'sklearn make_moons (2 features, 2 classes)',
        'circles': 'sklearn make_circles (2 features, 2 classes)',
        'blobs': 'sklearn make_blobs (6 features, 5 classes)',
        'mnist': 'MNIST handwritten digits (28×28 grayscale, 10 classes)',
        'cifar10': 'CIFAR-10 natural images (32×32 RGB, 10 classes)',
        'text': 'synthetic character sequences (20 chars, 10 classes)',
        'line': 'linear regression y=2x+1+noise (1 feature, 1 target)',
    }
    return infos.get(dataset, f'{dataset} dataset')


# ── train.py generator ──

def gen_enhanced_train(tier: str, model_type: str, class_name: str) -> str:
    """Generate train.py with argparse + improved training display."""
    _, train_code, _ = get_templates(tier, model_type)
    train_code = train_code.replace(
        "from config import *\nfrom model import {cn}\nfrom data import",
        _ARGPARSE_TRAIN + "from model import {cn}\nfrom data import"
    )
    train_code = train_code.replace('{cn}', class_name)

    # Post-process: enhance training display with progress bar + epoch counter
    train_code = _enhance_training_display(train_code)

    # Move model & data to the configured DEVICE (from config DEVICE_PRIORITY)
    train_code = _inject_device_support(train_code)

    # Hold out a validation split; track val metrics in training_log
    train_code = _inject_val_split(train_code, model_type)

    return train_code


_DEV_HELPER = (
    "# ── Device helper: move each batch to the configured DEVICE ──\n"
    "def _dev(batch):\n"
    "    if isinstance(batch, (tuple, list)):\n"
    "        return tuple(b.to(DEVICE) if torch.is_tensor(b) else b for b in batch)\n"
    "    return batch.to(DEVICE)\n"
)


def _inject_device_support(code: str) -> str:
    """Rewrite training code so model & data move to the configured DEVICE.

    Runs AFTER _enhance_training_display (loops are already enumerate()-based).
    Data loaders are wrapped with the _dev helper; generator expressions used
    for validation use map(_dev, lo). Returns code unchanged where no pattern
    matches (e.g. production tier already handles DEVICE explicitly).
    """
    # 1. Move model to DEVICE: m=M1();total_params → m=M1().to(DEVICE);total_params
    code = re.sub(r'\bm=(\w+)\(\);total_params', r'm=\1().to(DEVICE);total_params', code)

    # 2. Move every data batch to DEVICE (training & validation loops)
    code = code.replace('for i,(x,y) in enumerate(lo):',
                        'for i,(x,y) in enumerate(map(_dev, lo)):')
    code = code.replace('for i,(x1,x2) in enumerate(lo):',
                        'for i,(x1,x2) in enumerate(map(_dev, lo)):')
    code = code.replace('for i,(x1,x2,y) in enumerate(lo):',
                        'for i,(x1,x2,y) in enumerate(map(_dev, lo)):')
    code = code.replace('for i,(x,y1,y2) in enumerate(lo):',
                        'for i,(x,y1,y2) in enumerate(map(_dev, lo)):')
    code = code.replace('for i,x in enumerate(lo):',
                        'for i,x in enumerate(map(_dev, lo)):')
    code = code.replace('for i,real in enumerate(lo):',
                        'for i,real in enumerate(map(_dev, lo)):')
    # Validation generator expressions: sum(... for x,y in lo)
    code = code.replace('for x,y in lo)', 'for x,y in map(_dev, lo))')
    code = code.replace('for x in lo)', 'for x in map(_dev, lo))')

    # 3. GAN noise must be created on DEVICE
    code = code.replace('z=torch.randn(real.size(0),INPUT_DIM)',
                        'z=torch.randn(real.size(0),INPUT_DIM,device=DEVICE)')

    # 4. Contrastive: boolean mask must live on the same device as s
    code = code.replace('mask=torch.eye(len(h),dtype=torch.bool)',
                        'mask=torch.eye(len(h),dtype=torch.bool).to(DEVICE)')

    # 5. GCN: graph tensors must live on DEVICE
    code = code.replace('X=torch.stack([ds[i][0] for i in range(len(ds))])',
                        'X=torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)')
    code = code.replace('y=torch.tensor([ds[i][1] for i in range(len(ds))])',
                        'y=torch.tensor([ds[i][1] for i in range(len(ds))]).to(DEVICE)')
    code = code.replace('adj=build_adj(len(ds),ds.edges)',
                        'adj=build_adj(len(ds),ds.edges).to(DEVICE)')

    # 6. Insert the _dev helper after the data import
    code = code.replace('from data import SynData\n',
                        'from data import SynData\n' + _DEV_HELPER)
    return code


_VAL_SPLIT_CODE = (
    "ds=SynData()\n"
    "_n_val=max(1,int(len(ds)*VAL_SPLIT));_n_train=len(ds)-_n_val\n"
    "_train_ds,_val_ds=torch.utils.data.random_split(ds,[_n_train,_n_val],generator=torch.Generator().manual_seed(SEED))\n"
    "lo=torch.utils.data.DataLoader(_train_ds,BATCH_SIZE,shuffle=True)\n"
    "val_lo=torch.utils.data.DataLoader(_val_ds,BATCH_SIZE,shuffle=False)\n"
)

# Per-model-type validation metric blocks (inserted before history.append).
# Variables are prefixed with _ to avoid clashing with template locals.
_VAL_CLS = (
    "    with torch.no_grad():\n"
    "        _vl=0;_vc=0;_vn=0\n"
    "        for x,y in map(_dev, val_lo):\n"
    "            _vl+=criterion(m(x),y).item()*x.size(0)\n"
    "            _vc+=(m(x).argmax(1)==y).sum().item()\n"
    "            _vn+=x.size(0)\n"
    "        val_loss=_vl/_vn;val_acc=_vc/_vn\n"
    "    val_history.append((val_loss,val_acc))\n"
)

_VAL_REG = (
    "    with torch.no_grad():\n"
    "        _vl=0;_vn=0\n"
    "        for x,y in map(_dev, val_lo):\n"
    "            _vl+=criterion(m(x),y).item()*x.size(0)\n"
    "            _vn+=x.size(0)\n"
    "        val_loss=_vl/_vn\n"
    "    val_history.append((val_loss,))\n"
)

_VAL_AE = (
    "    with torch.no_grad():\n"
    "        _vl=0;_vn=0\n"
    "        for x in map(_dev, val_lo):\n"
    "            _vl+=nn.MSELoss()(m(x)[0],x).item()*x.size(0)\n"
    "            _vn+=x.size(0)\n"
    "        val_loss=_vl/_vn\n"
    "    val_history.append((val_loss,))\n"
)

_VAL_MT = (
    "    with torch.no_grad():\n"
    "        _vl=0;_vc1=0;_vc2=0;_vn=0\n"
    "        for x,y1,y2 in map(_dev, val_lo):\n"
    "            o1,o2=m(x)\n"
    "            _vl+=nn.CrossEntropyLoss()(o1,y1).item()*x.size(0)+nn.CrossEntropyLoss()(o2,y2).item()*x.size(0)\n"
    "            _vc1+=(o1.argmax(1)==y1).sum().item()\n"
    "            _vc2+=(o2.argmax(1)==y2).sum().item()\n"
    "            _vn+=x.size(0)\n"
    "        val_loss=_vl/_vn;val_acc=(_vc1+_vc2)/(2*_vn)\n"
    "    val_history.append((val_loss,val_acc))\n"
)


def _inject_val_split(code: str, model_type: str) -> str:
    """Add a validation split + per-epoch val metrics to training code.

    - GCN (whole-graph training) and GAN/contrastive/siamese (no simple
      criterion at epoch end) keep their original single-metric logging.
    - New models log extra 'Val Loss' (and 'Val Acc' for classifiers)
      columns in training_log.md; managers prefer them over train metrics.
    """
    if 'DataLoader' not in code:
        # GCN: whole-graph training, no batch split — but log footers
        # reference val_history, so still initialize it (stays empty).
        return code.replace("history=[]\n", "history=[]\nval_history=[]\n")

    # 1. Split the dataset into train/val loaders
    old = "ds=SynData();lo=torch.utils.data.DataLoader(ds,BATCH_SIZE,shuffle=True)\n"
    if old not in code:
        return code
    code = code.replace(old, _VAL_SPLIT_CODE)

    # 2. val_history accumulator
    code = code.replace("history=[]\n", "history=[]\nval_history=[]\n")

    # 3. Per-epoch val metrics before history.append
    if model_type in ('ce', 'cnn', 'rnn'):
        code = code.replace("    history.append((e,loss_val,acc))\n", _VAL_CLS + "    history.append((e,loss_val,acc))\n")
    elif model_type == 'mse':
        code = code.replace("    history.append((e,loss_val))\n", _VAL_REG + "    history.append((e,loss_val))\n")
    elif model_type in ('ae', 'vae'):
        code = code.replace("    history.append((e,recon_loss))\n", _VAL_AE + "    history.append((e,recon_loss))\n")
    elif model_type == 'mt':
        code = code.replace("    history.append((e,loss_val,a1,a2))\n", _VAL_MT + "    history.append((e,loss_val,a1,a2))\n")
    return code


def _inject_device_support_eval(code: str) -> str:
    """Rewrite eval code so the model & inputs use the configured DEVICE."""
    # 1. Model to DEVICE: m=M1();m.load_state_dict(...) → m=M1().to(DEVICE);...
    code = re.sub(r'\bm=(\w+)\(\);m\.load_state_dict',
                  r'm=\1().to(DEVICE);m.load_state_dict', code)

    # 2. Inputs to DEVICE
    code = code.replace('torch.from_numpy(X)', 'torch.from_numpy(X).to(DEVICE)')
    code = code.replace('torch.from_numpy(img.reshape(1,INPUT_DIM,8,8).astype(np.float32))',
                        'torch.from_numpy(img.reshape(1,INPUT_DIM,8,8).astype(np.float32)).to(DEVICE)')
    code = code.replace('torch.from_numpy(t)', 'torch.from_numpy(t).to(DEVICE)')
    code = code.replace('z=torch.randn(10,INPUT_DIM)', 'z=torch.randn(10,INPUT_DIM,device=DEVICE)')
    code = code.replace('x=torch.randn(10,INPUT_DIM)', 'x=torch.randn(10,INPUT_DIM,device=DEVICE)')
    code = code.replace('x1=torch.randn(5,INPUT_DIM)', 'x1=torch.randn(5,INPUT_DIM,device=DEVICE)')

    # 3. Outputs back to CPU before .numpy()
    code = code.replace('.argmax(1).numpy()', '.argmax(1).cpu().numpy()')
    code = code.replace(').numpy()', ').cpu().numpy()')
    code = code.replace('r.numpy()', 'r.cpu().numpy()')

    # 4. GCN graph tensors
    code = code.replace('X=torch.stack([ds[i][0] for i in range(len(ds))])',
                        'X=torch.stack([ds[i][0] for i in range(len(ds))]).to(DEVICE)')
    code = code.replace('y=torch.tensor([ds[i][1] for i in range(len(ds))])',
                        'y=torch.tensor([ds[i][1] for i in range(len(ds))]).to(DEVICE)')
    code = code.replace('adj=build_adj(len(ds),ds.edges)',
                        'adj=build_adj(len(ds),ds.edges).to(DEVICE)')
    return code


def _enhance_training_display(code: str) -> str:
    """Inject progress bar and improved formatting into training loop."""
    import re

    # 1. Add total_batches before epoch loop (skip GCN: no DataLoader/lo)
    if 'DataLoader' in code:
        code = code.replace(
            'for e in range(EPOCHS):',
            'total_batches=len(lo)\nfor e in range(EPOCHS):'
        )

    # 2. Convert 'for x,y in lo:' to 'for i,(x,y) in enumerate(lo):'
    #    and add progress bar update after o.step()
    progress_snippet = (
        '        if i%max(1,total_batches//20)==0:'
        '\n            pct=(i+1)/total_batches*100'
        '\n            bar="#"*int(pct//5)+"-"*(20-int(pct//5))'
        '\n            print(f"\\r  Epoch {e+1:3d}/{EPOCHS} [{bar}] {pct:5.1f}%",end="",flush=True)'
    )

    # Pattern: for x,y in lo:\n        l=...\n        o.zero_grad();l.backward();o.step()
    # Replace with: for i,(x,y) in enumerate(lo):\n        l=...\n        o.zero_grad();l.backward();o.step()\n        <progress>
    code = re.sub(
        r'for (\w+),(\w+) in lo:',
        r'for i,(\1,\2) in enumerate(lo):',
        code
    )
    # Handle single-input case: for x in lo:
    code = re.sub(
        r'for (\w+) in lo:',
        r'for i,\1 in enumerate(lo):',
        code
    )
    # Handle triple-input case: for x1,x2,y in lo:
    code = re.sub(
        r'for (\w+),(\w+),(\w+) in lo:',
        r'for i,(\1,\2,\3) in enumerate(lo):',
        code
    )

    # Insert progress bar after o.step() (or od.step() for GAN).
    # Only for DataLoader loops (skip GCN: no per-batch loop, no i)
    if 'enumerate(lo)' in code:
        code = re.sub(
            r'(\bo\.step\(\))',
            r'\1\n' + progress_snippet.replace('\\', '\\\\'),
            code
        )
        # GAN: od.step() and og.step() — add progress after od.step()
        code = re.sub(
            r'(\bod\.step\(\))',
            r'\1\n' + progress_snippet.replace('\\', '\\\\'),
            code
        )

    # 3. Improve epoch print format (overwrite progress bar with \r)
    code = re.sub(
        r"print\(f'Epoch \{e:3d\}: loss=\{loss_val:\.4f\}, acc=\{acc:\.4f\}'\)",
        r"print(f'\\r  Epoch {e+1:3d}/{EPOCHS} | loss={loss_val:.4f}  acc={acc:.4f}          ')",
        code
    )
    code = re.sub(
        r"print\(f'Epoch \{e:3d\}: loss=\{loss_val:\.4f\}'\)",
        r"print(f'\\r  Epoch {e+1:3d}/{EPOCHS} | loss={loss_val:.4f}          ')",
        code
    )
    code = re.sub(
        r"print\(f'Epoch \{e:3d\}: recon_loss=\{recon_loss:\.4f\}'\)",
        r"print(f'\\r  Epoch {e+1:3d}/{EPOCHS} | recon_loss={recon_loss:.4f}          ')",
        code
    )
    code = re.sub(
        r"print\(f'Epoch \{e:3d\}: d_loss=\{dl:\.4f\}, g_loss=\{gl:\.4f\}'\)",
        r"print(f'\\r  Epoch {e+1:3d}/{EPOCHS} | d_loss={dl:.4f}  g_loss={gl:.4f}          ')",
        code
    )
    code = re.sub(
        r"print\(f'Epoch \{e:3d\}: loss=\{loss_val:\.4f\}, acc1=\{a1:\.4f\}, acc2=\{a2:\.4f\}'\)",
        r"print(f'\\r  Epoch {e+1:3d}/{EPOCHS} | loss={loss_val:.4f}  acc1={a1:.4f}  acc2={a2:.4f}          ')",
        code
    )

    # 4. Improve model info header
    code = code.replace(
        "print(f'Model: {total_params} parameters')",
        "print(f'Model: {total_params} parameters | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}\\n')"
    )

    # 5. Replace hardcoded optimizer with config-driven
    code = code.replace(
        "o=torch.optim.Adam(m.parameters(),lr=LR)",
        (
            "if OPTIMIZER=='sgd':\n"
            "    o=torch.optim.SGD(m.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)\n"
            "elif OPTIMIZER=='adamw':\n"
            "    o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)\n"
            "else:\n"
            "    o=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)"
        )
    )
    # Replace scheduler placeholder (Standard tier already has scheduler, Quick doesn't)
    code = code.replace(
        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',patience=5,factor=0.5)",
        (
            "if SCHEDULER=='cosine':\n"
            "    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=EPOCHS)\n"
            "elif SCHEDULER=='step':\n"
            "    scheduler=torch.optim.lr_scheduler.StepLR(o,step_size=max(1,EPOCHS//3),gamma=0.5)\n"
            "elif SCHEDULER=='plateau':\n"
            "    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',patience=max(1,PATIENCE//2),factor=0.5)\n"
            "else:\n"
            "    scheduler=None"
        )
    )
    # Replace hardcoded patience with config value
    code = code.replace('PATIENCE=10', 'PATIENCE=PATIENCE')
    # Replace hardcoded grad clip
    code = code.replace(
        "nn.utils.clip_grad_norm_(m.parameters(),1.0)",
        "nn.utils.clip_grad_norm_(m.parameters(),GRAD_CLIP)"
    )
    # Add seed setting at the top
    code = code.replace(
        'ds=SynData()',
        'torch.manual_seed(SEED)\nds=SynData()'
    )

    # 6. Add resume-from-checkpoint support
    resume_snippet = (
        "start_epoch=0\n"
        "if _a.resume is not None:\n"
        "    ckpt=torch.load(_a.resume,weights_only=False,map_location='cpu')\n"
        "    m.load_state_dict(ckpt['model'])\n"
        "    o.load_state_dict(ckpt.get('opt',ckpt.get('optimizer',{})))\n"
        "    start_epoch=ckpt.get('epoch',-1)+1\n"
        "    print(f'Resumed from {_a.resume} (epoch {start_epoch})')\n"
        "EPOCHS=max(EPOCHS,start_epoch+1)\n"
    )
    code = code.replace(
        "for e in range(EPOCHS):",
        resume_snippet + "for e in range(start_epoch, EPOCHS):"
    )

    return code


# ── predict.py generator ──

def gen_predict(class_name: str, input_dim: int, model_type: str) -> str:
    """Generate predict.py for inference demonstration."""
    if model_type == 'cnn':
        body = (
            "img = np.zeros((8, 8), dtype=np.float32)\n"
            "img[3:5, :] = 1.0\n"
            f"x = torch.from_numpy(img.reshape(1, {input_dim}, 8, 8).astype(np.float32)).to(DEVICE)\n"
            "if INPUT_DIM == 3: x = x.repeat(1, 3, 1, 1)\n"
            "print(f'Input shape: {x.shape}')"
        )
    elif model_type == 'rnn':
        body = (
            "x = torch.randn(1, 15, 1, device=DEVICE)\n"
            "print(f'Input shape: {x.shape}')"
        )
    elif model_type == 'gan':
        body = (
            f"z = torch.randn(1, {input_dim}, device=DEVICE)\n"
            "print(f'Latent z shape: {z.shape}')\n"
            "x = z\n"
        )
    elif model_type == 'gcn':
        body = (
            f"x = torch.randn(10, {input_dim}, device=DEVICE)\n"
            "adj = torch.eye(10, device=DEVICE)\n"
            "print(f'Input shape: {x.shape}, adj: {adj.shape}')"
        )
    else:
        body = f"x = torch.randn(1, {input_dim}, device=DEVICE)\nprint(f'Input shape: {{x.shape}}')"

    call = 'model(x, adj)' if model_type == 'gcn' else 'model(x)'

    return (
        "import torch\n"
        "import numpy as np\n"
        f"from model import {class_name}\n"
        f"from config import *\n\n"
        f"model = {class_name}().to(DEVICE)\n"
        f"model.load_state_dict(torch.load('model.pth', weights_only=True))\n"
        f"model.eval()\n\n"
        + body +
        f"\n\nwith torch.no_grad():\n"
        f"    out = {call}\n"
        "    if isinstance(out, tuple): out = out[0]\n"
        "    if out.numel() == 1:\n"
        "        print(f'Prediction: {out.item():.4f}')\n"
        "    else:\n"
        "        print(f'Prediction shape: {out.shape}')\n"
        "print('Done.')\n"
    )


# ── visualize.py generator ──

def gen_visualize(model_type: str) -> str:
    """Generate visualize.py — reads training_log.md, plots REAL data."""
    meta = _ARCH_META.get(model_type, _ARCH_META['ce'])

    # Determine which columns to expect based on model type
    if model_type == 'gan':
        cols = "['epoch', 'd_loss', 'g_loss']"
        y_axes = "[('D Loss', 'd_loss', 'tab:red'), ('G Loss', 'g_loss', 'tab:blue')]"
    elif model_type == 'mt':
        cols = "['epoch', 'loss', 'acc1', 'acc2']"
        y_axes = "[('Loss', 'loss', 'tab:red'), ('Acc1', 'acc1', 'tab:blue'), ('Acc2', 'acc2', 'tab:green')]"
    elif model_type in ('ae', 'vae'):
        cols = "['epoch', 'recon_loss']"
        y_axes = "[('Reconstruction Loss', 'recon_loss', 'tab:blue')]"
    elif model_type == 'mse':
        cols = "['epoch', 'loss']"
        y_axes = "[('Loss (MSE)', 'loss', 'tab:red')]"
    else:
        cols = "['epoch', 'loss', 'accuracy']"
        y_axes = "[('Loss', 'loss', 'tab:red'), ('Accuracy', 'accuracy', 'tab:green')]"

    return (
        '"""Visualize training progress — reads training_log.md and plots real curves."""\n'
        'import re\n'
        'import matplotlib\n'
        'import matplotlib.pyplot as plt\n'
        "matplotlib.use('Agg')\n"
        '\n'
        '# ── Parse training_log.md ──\n'
        'try:\n'
        "    with open('training_log.md', 'r') as f:\n"
        '        text = f.read()\n'
        'except FileNotFoundError:\n'
        '    print("No training_log.md found. Train first with: python train.py")\n'
        '    exit(1)\n'
        '\n'
        '# Extract table rows\n'
        f'expected_cols = {cols}\n'
        'rows = []\n'
        'for line in text.split("\\n"):\n'
        '    parts = [p.strip() for p in line.split("|")[1:-1]]\n'
        '    if len(parts) >= 2 and parts[0].isdigit():\n'
        '        try:\n'
        '            row = [float(p) if "." in p or p.replace("-","").isdigit() else None for p in parts]\n'
        '            row[0] = int(parts[0])\n'
        '            rows.append(row)\n'
        '        except ValueError:\n'
        '            continue\n'
        '\n'
        'if not rows:\n'
        '    print("No training data found in training_log.md")\n'
        '    exit(1)\n'
        '\n'
        '# Build data dict\n'
        'data = {}\n'
        'for i, col in enumerate(expected_cols):\n'
        '    if i < len(rows[0]):\n'
        f'        data[col] = [r[i] for r in rows if r[i] is not None]\n'
        '\n'
        '# ── Plot ──\n'
        f'y_axes = {y_axes}\n'
        'n_plots = len(y_axes)\n'
        'fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)\n'
        'if n_plots == 1:\n'
        '    axes = [axes]\n'
        '\n'
        'for ax, (label, key, color) in zip(axes, y_axes):\n'
        '    if key in data and len(data[key]) == len(data.get("epoch", [])):\n'
        '        ax.plot(data["epoch"], data[key], "o-", color=color, markersize=3, label=label)\n'
        '        ax.set_ylabel(label, color=color)\n'
        '        ax.legend(loc="upper right")\n'
        '        ax.grid(True, alpha=0.3)\n'
        '    else:\n'
        '        ax.text(0.5, 0.5, f"No data for {label}", ha="center", va="center", transform=ax.transAxes)\n'
        '\n'
        'axes[-1].set_xlabel("Epoch")\n'
        'fig.suptitle("Training Progress", fontsize=14, fontweight="bold")\n'
        'plt.tight_layout()\n'
        'plt.savefig("training_curve.png", dpi=150)\n'
        'print("Saved training_curve.png")\n'
    )


# ── data_explore.py generator (NEW) ──

def gen_data_explore(class_name: str) -> str:
    """Generate data_explore.py — lets users inspect the dataset."""
    lines = [
        '"""Explore the dataset — load it and print statistics."""',
        'import torch',
        'from config import *',
        'from data import SynData',
        '',
        'print("=" * 50)',
        'print("  DATA EXPLORER")',
        'print("=" * 50)',
        '',
        'ds = SynData()',
        'n = len(ds)',
        'print(f"Dataset:    {DATASET}")',
        'print(f"Samples:    {n}")',
        'print(f"Input dim:  {INPUT_DIM}")',
        'print(f"Output dim: {OUTPUT_DIM}")',
        '',
        '# Inspect first sample',
        'sample = ds[0]',
        'if isinstance(sample, (tuple, list)):',
        '    sx = sample[0]',
        '    shape_str = str(sx.shape) if hasattr(sx, "shape") else type(sx).__name__',
        '    print(f"Sample[0] (x) shape: {shape_str}")',
        '    for i, s in enumerate(sample[1:], 1):',
        '        val = s.item() if hasattr(s, "item") else s',
        '        print(f"Sample[{i}] (y/target): {val}")',
        'else:',
        '    shape_str = str(sample.shape) if hasattr(sample, "shape") else type(sample).__name__',
        '    print(f"Sample shape: {shape_str}")',
        '',
        '# Batch stats',
        'loader = torch.utils.data.DataLoader(ds, batch_size=min(256, n), shuffle=False)',
        'batch = next(iter(loader))',
        'xb = batch[0] if isinstance(batch, (tuple, list)) else batch',
        'print(f"\\nBatch x stats — mean: {xb.float().mean():.3f}, std: {xb.float().std():.3f}")',
        'print(f"Batch x range: [{xb.float().min():.3f}, {xb.float().max():.3f}]")',
        'print("\\nDone.")',
    ]
    return '\n'.join(lines) + '\n'


# ── requirements.txt generator ──

def gen_requirements() -> str:
    """Generate requirements.txt."""
    return "torch>=2.0.0\nnumpy\nmatplotlib\n# scikit-learn (optional, for real datasets)\n"


# ── README generator ──

def gen_readme(description: str, params: int, model_type: str,
               input_dim: int, output_dim: int, dataset: str, tier: str = 'quick') -> str:
    """Generate architecture-specific README.md."""
    meta = _ARCH_META.get(model_type, _ARCH_META['ce'])

    tier_desc = {'quick': 'Quick (basic)', 'standard': 'Standard (lr scheduler, early stopping, checkpoints)',
                 'production': 'Production (DDP, AMP, grad accumulation)'}
    tier_files = {
        'quick': 'predict.py, visualize.py',
        'standard': 'predict.py, visualize.py (real data), sweep.py, checkpoints/',
        'production': 'model/ sub-package, configs/, scripts/ (benchmark, profile, export), checkpoints/, logs/'
    }

    return (
        f"# {description}\n\n"
        f"- **Task**: {meta['task']}\n"
        f"- **Parameters**: {params:,}\n"
        f"- **Tier**: {tier_desc.get(tier, tier)}\n"
        f"- **Input**: {input_dim}-dim {meta['input_desc']}\n"
        f"- **Output**: {meta['output_desc']}\n\n"
        f"---\n\n"
        f"## Data\n\n"
        f"Dataset source is defined in `config.py` → `DATASET = '{dataset}'`.\n"
        f"The data loader lives in `data.py` (class `SynData`).\n\n"
        f"To inspect the data:\n"
        f"```bash\n"
        f"python data_explore.py\n"
        f"```\n\n"
        f"---\n\n"
        f"## Files\n\n"
        f"| File | Purpose |\n"
        f"|------|--------|\n"
        f"| `config.py` | All hyperparameters: LR, epochs, batch size, data config |\n"
        f"| `model.py` | PyTorch nn.Module — the neural network architecture |\n"
        f"| `data.py` | Dataset class — loads and preprocesses training data |\n"
        f"| `train.py` | Training loop — `{meta['loss']}`, Adam optimizer |\n"
        f"| `eval.py` | Evaluation — reports {meta['metric']} |\n"
        f"| `data_explore.py` | **Inspect the dataset** — print stats & shapes |\n"
        f"| `requirements.txt` | Python dependencies |\n"
        f"| *+ tier extras* | {tier_files.get(tier, '')} |\n\n"
        f"---\n\n"
        f"## Run\n\n"
        f"```bash\n"
        f"# 1. Explore the data first\n"
        f"python data_explore.py\n\n"
        f"# 2. Train\n"
        f"python train.py --epochs 50 --lr 0.001 --batch-size 128\n\n"
        f"# 3. Evaluate\n"
        f"python eval.py\n"
        f"```\n"
    )


# ── Main folder generator ──

def gen_folder(base_dir: str, index: int, description: str, code: str,
               class_name_template: str, params: int, input_dim: int,
               output_dim: int, model_type: str, dataset: str = "syn",
               device_priority: Optional[list] = None) -> str:
    """Generate a single model folder with tier-appropriate files.

    Tier is determined by param count:
      - < 50K:       quick (basic scripts)
      - 50K ~ 50M:   standard (+ sweep, visualize, checkpoints)
      - > 50M:       production (+ DDP, AMP, model sub-package, scripts)
    """
    tier = _get_tier(params)

    folder_name = f"{index:03d}-{description}"
    folder = os.path.join(base_dir, folder_name)
    class_name = class_name_template.format(index)
    model_code = code.format(index)

    # Determine loss type
    loss_type = model_type if model_type in ('ce', 'mse', 'ae', 'contrastive') else 'ce'

    # --- Resolve dataset first (may override dims, model type, data code) ---
    ds_code, ds_input_dim, ds_output_dim = get_dataset_code(dataset, input_dim, output_dim)
    if ds_code and dataset != 'syn':
        actual_data_code = ds_code
        # Override model type to match dataset task
        if dataset == 'line':
            model_type = 'mse'
        elif dataset in ('iris', 'wine', 'breast_cancer', 'moons', 'circles',
                         'blobs', 'mnist', 'cifar10', 'text'):
            if model_type not in ('ae', 'vae', 'gan', 'contrastive', 'siamese', 'mt'):
                model_type = 'ce'
        # Rewrite model code to use dataset's actual dimensions
        model_code = _rewrite_model_dims(model_code, input_dim, output_dim,
                                         ds_input_dim, ds_output_dim, model_type)
        input_dim, output_dim = ds_input_dim, ds_output_dim
    else:
        raw_data_code, _, _ = get_templates(tier, model_type)
        actual_data_code = raw_data_code

    # --- config.py (after dims are final) ---
    write_file(os.path.join(folder, "config.py"),
               gen_config(input_dim, output_dim, loss_type, dataset, model_type,
                          device_priority))

    # --- model.py ---
    write_file(os.path.join(folder, "model.py"),
               "import torch\nimport torch.nn as nn\n\n" + model_code)

    # --- data.py ---
    write_file(os.path.join(folder, "data.py"),
               actual_data_code.replace('{cn}', class_name)
                              .replace('{inp}', str(input_dim))
                              .replace('{outp}', str(output_dim)))

    # --- train.py ---
    write_file(os.path.join(folder, "train.py"),
               gen_enhanced_train(tier, model_type, class_name))

    # --- eval.py ---
    _, _, raw_eval_code = get_templates(tier, model_type)
    eval_code = raw_eval_code.replace('{cn}', class_name).replace(
        "torch.load('model.pth')", "torch.load('model.pth', weights_only=True)")
    eval_code = _inject_device_support_eval(eval_code)
    write_file(os.path.join(folder, "eval.py"), eval_code)

    # --- data_explore.py ---
    write_file(os.path.join(folder, "data_explore.py"),
               gen_data_explore(class_name))

    # --- Tier-specific files ---
    if tier == 'quick':
        # Quick: predict + visualize generated inline
        write_file(os.path.join(folder, "predict.py"),
                   gen_predict(class_name, input_dim, model_type))
        write_file(os.path.join(folder, "visualize.py"), gen_visualize(model_type))

    elif tier == 'standard':
        # Standard: predict/visualize/sweep from templates
        extras = get_extra_files(tier, model_type, class_name)
        for fname, content in extras.items():
            write_file(os.path.join(folder, fname), content)

    elif tier == 'production':
        # Production: model sub-package, configs/, scripts/
        extras = get_extra_files(tier, model_type, class_name)
        for fname, content in extras.items():
            write_file(os.path.join(folder, fname), content)
        # Also include standard extras (predict, visualize, sweep)
        from .templates import get_extra_files as get_std_extras
        std_extras = get_std_extras('standard', model_type, class_name)
        for fname, content in std_extras.items():
            write_file(os.path.join(folder, fname), content)

    # --- requirements.txt ---
    write_file(os.path.join(folder, "requirements.txt"), gen_requirements())

    # --- README.md ---
    write_file(os.path.join(folder, "README.md"),
               gen_readme(description, params, model_type, input_dim, output_dim, dataset, tier))

    return folder
