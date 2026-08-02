"""Command-line interface for NetGen — generate + manage models."""
import argparse
import os
import sys
from typing import Optional

from .search import find_candidates, list_architectures
from .generator import (gen_folder, IMAGE_2D_ARCHS, IMAGE_DATASETS,
                          IMAGE_PATCH_ARCHS)
from .manager import (
    list_models, info_model, compare_models, clean_models, export_models,
    benchmark_models, train_model, eval_model, sweep_model
)


# ── Shared constants ──

_TYPE_LABELS = {
    'ce': 'cls', 'mse': 'reg', 'rnn': 'seq',
    'cnn': 'img', 'ae': 'ae', 'vae': 'vae',
    'mt': 'mt', 'gan': 'gan', 'contrastive': 'ctr',
    'siamese': 'sim', 'gcn': 'gcn',
}

_DATASET_TASK = {
    'iris':          'classification',
    'wine':          'classification',
    'breast_cancer': 'classification',
    'moons':         'classification',
    'circles':       'classification',
    'blobs':         'classification',
    'mnist':         'classification',
    'cifar10':       'classification',
    'text':          'classification',
    'line':          'regression',
}

# Vector datasets: every supervised architecture that consumes 1-D feature
# vectors (x, y). Excluded:
#  - CNN-style (2-D input), patch (vit/mixer), token (gpt/t5), graph (gcn)
#  - RNN family (their SynData is a sine sequence; forward expects 3-D inputs)
#  - self-supervised/generative (ae/sae/vae/contrastive/siamese) and
#    multitask (mt): their training templates need unlabeled or
#    multi-target data, which real labeled datasets don't provide.
_VECTOR_DATASET_ARCHS = sorted(
    set(list_architectures()) - IMAGE_2D_ARCHS - IMAGE_PATCH_ARCHS
    - {'gpt', 't5', 'gcn', 'gan', 'lstm', 'gru', 'bilstm', 'rnn', 'attnlstm',
       'ae', 'sae', 'vae', 'contrastive', 'siamese', 'multitask',
       'selfattn', 'transformer', 'highway'})

# cifar10: 2-D nets + patch nets keep the image shape, everything else
# (except token/graph/gan) gets flattened samples.
_IMAGE_DATASET_ARCHS = sorted(set(_VECTOR_DATASET_ARCHS)
                              | set(IMAGE_2D_ARCHS) | set(IMAGE_PATCH_ARCHS))
# mnist is grayscale: patch nets (hard-coded RGB) can't be used.
_MNIST_ARCHS = sorted(set(_VECTOR_DATASET_ARCHS) | set(IMAGE_2D_ARCHS))

_DATASET_ARCHS = {
    'iris':          _VECTOR_DATASET_ARCHS,
    'wine':          _VECTOR_DATASET_ARCHS,
    'breast_cancer': _VECTOR_DATASET_ARCHS,
    'moons':         _VECTOR_DATASET_ARCHS,
    'circles':       _VECTOR_DATASET_ARCHS,
    'blobs':         _VECTOR_DATASET_ARCHS,
    'line':          _VECTOR_DATASET_ARCHS,
    'mnist':         _MNIST_ARCHS,
    'cifar10':       _IMAGE_DATASET_ARCHS,
    'text':          ['gpt', 't5'],
}

_PRESETS = {
    'cv':    ['cnn', 'rescnn', 'sepcnn', 'densecnn', 'vit', 'unet', 'mixer'],
    'nlp':   ['lstm', 'gru', 'bilstm', 'attnlstm', 'transformer', 'selfattn', 'gpt', 't5'],
    'gen':   ['ae', 'sae', 'vae', 'gan'],
    'light': ['unary', 'linear', 'mlp'],
    'all':   None,  # None = use all architectures
}


# ── Helpers ──

def _parse_range(range_str: str) -> tuple[int, int]:
    try:
        lo_str, hi_str = range_str.split("-")
        lo, hi = int(lo_str), int(hi_str)
    except ValueError:
        raise ValueError(f"Invalid range format: '{range_str}'. Expected e.g. '10000-20000'.")
    if lo < 0 or hi < 0:
        raise ValueError("Parameter range bounds must be non-negative.")
    if lo >= hi:
        raise ValueError(f"Low bound ({lo}) must be less than high bound ({hi}).")
    return lo, hi


_VALID_DEVICES = ('cuda', 'mps', 'cpu')


def _parse_device(device_str: Optional[str]) -> list[str]:
    """Parse --device 'cuda,mps' into a priority list.

    'cpu' is ALWAYS the final fallback, so it may be omitted from the list
    (e.g. 'cuda,cpu' == 'cuda'). Empty -> default ['cuda', 'mps'].
    """
    if not device_str:
        return ['cuda', 'mps']
    devices = [d.strip().lower() for d in device_str.split(',') if d.strip()]
    for d in devices:
        if d not in _VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{d}'. Supported: {', '.join(_VALID_DEVICES)}.")
    return devices


def _resolve_arch_filter(opts, dataset: str) -> Optional[list[str]]:
    """Resolve architecture filter from --arch or --preset, plus dataset compatibility."""
    arch_filter = None

    # --preset takes priority
    if hasattr(opts, 'preset') and opts.preset and opts.preset != 'all':
        if opts.preset not in _PRESETS:
            print(f"Error: Unknown preset '{opts.preset}'. Choose: {list(_PRESETS.keys())}", file=sys.stderr)
            return None
        arch_filter = list(_PRESETS[opts.preset])

    # --arch refines further (if preset also set, intersect)
    if opts.arch:
        user_archs = [a.strip() for a in opts.arch.split(",") if a.strip()]
        valid = list_architectures()
        invalid = [a for a in user_archs if a not in valid]
        if invalid:
            print(f"Warning: Unknown architectures ignored: {invalid}", file=sys.stderr)
        user_archs = [a for a in user_archs if a in valid]
        if arch_filter:
            arch_filter = [a for a in arch_filter if a in user_archs]
        else:
            arch_filter = user_archs
        if not arch_filter:
            print("Error: No architectures match both --preset and --arch.", file=sys.stderr)
            return None

    # Dataset compatibility filter
    if dataset != 'syn' and dataset in _DATASET_ARCHS:
        compat = _DATASET_ARCHS[dataset]
        if arch_filter:
            bad = [a for a in arch_filter if a not in compat]
            if bad:
                print(f"Warning: {bad} incompatible with dataset '{dataset}'. "
                      f"Compatible: {compat}", file=sys.stderr)
            arch_filter = [a for a in arch_filter if a in compat]
            if not arch_filter:
                print(f"Error: No architectures compatible with dataset '{dataset}'. "
                      f"Compatible: {compat}", file=sys.stderr)
                return None
        else:
            arch_filter = compat

    return arch_filter


# ── Subcommand: generate ──

def _cmd_generate(args):
    try:
        lo, hi = _parse_range(args.range)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    arch_filter = _resolve_arch_filter(args, args.dataset)
    if arch_filter is None and args.arch:
        return 1

    try:
        device_priority = _parse_device(args.device)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    output_dir = os.path.abspath(args.output)

    print(f"\nNetGen - Generating {args.count} models in range {lo:,}-{hi:,}")
    if arch_filter:
        print(f"Architectures: {', '.join(arch_filter)}")
    print(f"Dataset: {args.dataset}")
    print(f"Device priority: {', '.join(device_priority)} (cpu is always the final fallback)")
    print(f"Output: {output_dir}\n")

    if args.dataset in IMAGE_DATASETS:
        # Image datasets fix the input shape per architecture family:
        # 2-D nets get C channels, patch nets keep RGB, vector nets get
        # the flattened image.
        c_in = 1 if args.dataset == 'mnist' else 3
        flat_dim = 784 if args.dataset == 'mnist' else 3072
        candidates = []
        two_d = [a for a in arch_filter if a in IMAGE_2D_ARCHS]
        patch = [a for a in arch_filter if a in IMAGE_PATCH_ARCHS]
        vec = [a for a in arch_filter
               if a not in IMAGE_2D_ARCHS and a not in IMAGE_PATCH_ARCHS]
        if two_d:
            candidates += find_candidates(lo, hi, args.count, args.seed, two_d,
                                          fixed_input=c_in)
        if patch:
            candidates += find_candidates(lo, hi, args.count, args.seed, patch)
        if vec:
            candidates += find_candidates(lo, hi, args.count, args.seed, vec,
                                          fixed_input=flat_dim)
        candidates.sort(key=lambda x: x[2])
    else:
        candidates = find_candidates(lo, hi, args.count, args.seed, arch_filter)
        candidates.sort(key=lambda x: x[2])

    if len(candidates) == 0:
        print(f"Warning: Could not find any architectures in range {lo:,}-{hi:,}", file=sys.stderr)
        print("Try widening the range or using --arch to select different architectures.", file=sys.stderr)
        return 1

    if len(candidates) < args.count:
        print(f"Note: Found {len(candidates)} candidates (requested {args.count}).")

    generated = 0
    tasks = [
        (output_dir, i + 1, desc, code, "M{}", params, input_dim, output_dim,
         model_type, args.dataset, device_priority)
        for i, (desc, code, params, input_dim, output_dim, model_type) in enumerate(candidates[:args.count])
    ]

    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(min(args.jobs, len(tasks))) as pool:
            folders = pool.starmap(gen_folder, tasks)
    else:
        folders = [gen_folder(*t) for t in tasks]

    for i, folder in enumerate(folders):
        label = _TYPE_LABELS.get(candidates[i][5], candidates[i][5])
        print(f"  [{i + 1:3d}] {os.path.basename(folder)}: {candidates[i][2]:>12,} params  [{label}]")
        generated += 1

    print(f"\nGenerated {generated} models in {output_dir}")
    print("Each folder: model.py, data.py, train.py, eval.py, predict.py,")
    print("           visualize.py, config.py, requirements.txt, README.md")
    print("To train: cd <folder> && python train.py")
    print("To manage: netgen list --dir " + output_dir)
    return 0


# ── Subcommand: list ──

def _cmd_list(args):
    print(list_models(args.dir))


# ── Subcommand: info ──

def _cmd_info(args):
    print(info_model(args.dir, args.id))


# ── Subcommand: compare ──

def _cmd_compare(args):
    reverse = args.sort in ('loss', 'recon_loss')
    print(compare_models(args.dir, args.sort, args.top, reverse))


# ── Subcommand: clean ──

def _cmd_clean(args):
    print(clean_models(args.dir, args.keep_best, args.status, args.untrained, args.dry_run))


# ── Subcommand: train ──

def _cmd_train(args):
    print(train_model(args.dir, args.id, args.epochs, args.lr,
                      args.batch_size, args.device, args.seed))


# ── Subcommand: eval ──

def _cmd_eval(args):
    print(eval_model(args.dir, args.id))


# ── Subcommand: sweep ──

def _cmd_sweep(args):
    lrs = [float(x) for x in args.lrs.split(',')] if args.lrs else None
    batches = [int(x) for x in args.batches.split(',')] if args.batches else None
    print(sweep_model(args.dir, args.id, args.epochs, lrs, batches,
                      args.device, args.seed))


# ── Subcommand: benchmark ──

def _cmd_benchmark(args):
    print(benchmark_models(args.dir, args.epochs, args.lr, args.batch_size,
                           args.seed, args.device, args.retries,
                           args.workers, args.force, args.time_budget))


# ── Subcommand: export ──

def _cmd_export(args):
    print(export_models(args.dir, args.format, args.output))


# ── Subcommand: archs ──

_ARCH_FAMILIES = {
    'Classification': ['mlp', 'deep', 'wide', 'resblock', 'highway', 'moe',
                       'transformer', 'selfattn', 'gpt', 't5', 'linear'],
    'Image (CNN)': ['cnn', 'rescnn', 'sepcnn', 'densecnn'],
    'Image (Modern)': ['vit', 'mixer', 'unet'],
    'Sequence': ['lstm', 'gru', 'bilstm', 'attnlstm'],
    'Generative': ['ae', 'sae', 'vae', 'gan'],
    'Similarity': ['siamese', 'contrastive'],
    'Graph': ['gcn'],
    'Multi-Task': ['multitask'],
    'Tiny (1-param)': ['unary'],
}

_ARCH_DESC = {
    'mlp': '3-layer Multi-Layer Perceptron',
    'deep': 'Deep MLP (N layers, same width)',
    'wide': 'Wide Net (one huge hidden layer)',
    'resblock': 'Residual blocks with skip connections',
    'highway': 'Highway network with gated connections',
    'moe': 'Mixture of Experts',
    'transformer': 'Transformer encoder stack',
    'selfattn': 'Pure self-attention (no FFN)',
    'gpt': 'GPT-style decoder transformer',
    't5': 'T5 encoder-decoder transformer',
    'linear': 'Single linear layer',
    'cnn': 'Small ConvNet (8x8 images)',
    'rescnn': 'Multi-stage residual CNN (ResNet style)',
    'sepcnn': 'Depthwise separable CNN (MobileNet style)',
    'densecnn': 'Densely connected CNN (DenseNet style)',
    'vit': 'Vision Transformer (patch embedding)',
    'mixer': 'MLP-Mixer (token + channel mixing)',
    'unet': 'U-Net with skip connections',
    'lstm': 'Long Short-Term Memory',
    'gru': 'Gated Recurrent Unit',
    'bilstm': 'Bidirectional LSTM',
    'attnlstm': 'LSTM + multi-head attention pooling',
    'ae': 'Autoencoder',
    'sae': 'Stacked Autoencoder',
    'vae': 'Variational Autoencoder',
    'gan': 'Generative Adversarial Network',
    'siamese': 'Siamese pairwise similarity',
    'contrastive': 'Contrastive learning embedding',
    'gcn': '2-layer Graph Convolutional Network',
    'multitask': 'Multi-task (shared trunk + heads)',
    'unary': '1-param (a:linear-no-bias, b:weight, c:bias)',
}


def _cmd_archs(args):
    if args.tree:
        lines = ["", "Architecture Family Tree", "========================", ""]
        for family, archs in _ARCH_FAMILIES.items():
            lines.append(family)
            for a in archs:
                lines.append(f"  +-- {a:<15s}  {_ARCH_DESC.get(a, '')}")
            lines.append("")
        lines.append(f"Total: {len(_ARCH_DESC)} architectures")
        print('\n'.join(lines))
    else:
        from .search import list_architectures
        archs = list_architectures()
        print(f"\n{'Architecture':<15s}  {'Description':<55s}")
        print(f"{'-'*15}  {'-'*55}")
        for a in archs:
            print(f"{a:<15s}  {_ARCH_DESC.get(a, ''):<55s}")
        print(f"\nTotal: {len(archs)} architectures")
    return 0


# ── Main entry ──

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netgen",
        description="NetGen - Batch Neural Network Model Generator & Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # netgen generate
    gen = sub.add_parser("generate", aliases=["gen"],
                         help="Generate model training folders",
                         description="Generate batch PyTorch model folders with target parameter counts.")
    gen.add_argument("--range", required=True, help="Parameter count range, e.g. '10000-20000'.")
    gen.add_argument("--count", type=int, default=20, help="Number of models (default: 20).")
    gen.add_argument("--output", "-o", default="./generated_models", help="Output directory.")
    gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    gen.add_argument("--dataset", default="syn",
                    choices=['syn', 'iris', 'wine', 'breast_cancer', 'moons',
                             'circles', 'blobs', 'mnist', 'cifar10', 'text', 'line'],
                    help="Dataset (default: syn). Real data: iris/wine/breast_cancer/"
                         "moons/circles/blobs/mnist/cifar10/text/line.")
    gen.add_argument("--arch", default=None, help="Architecture filter, e.g. 'mlp,cnn,lstm'.")
    gen.add_argument("--preset", "-p", default=None,
                     choices=['cv', 'nlp', 'gen', 'light', 'all'],
                     help="Preset filter: cv, nlp, gen, light, all.")
    gen.add_argument("--jobs", "-j", type=int, default=1,
                     help="Parallel workers for generation (default: 1).")
    gen.add_argument("--device", default=None,
                     help="Training device priority, e.g. 'cuda,mps' (default: cuda,mps). "
                          "'cpu' is always the final fallback and may be omitted, "
                          "e.g. 'cuda,cpu' == 'cuda'. Supported: cuda, mps, cpu.")

    # netgen list
    lst = sub.add_parser("list", aliases=["ls"],
                         help="List all generated models with status",
                         description="Scan a directory and list all model folders with training status.")
    lst.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")

    # netgen info
    inf = sub.add_parser("info", aliases=["inspect", "show"],
                         help="Show detailed info about a model",
                         description="Display full details for a specific model by ID or folder name.")
    inf.add_argument("id", help="Model ID (e.g. 001) or folder name.")
    inf.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")

    # netgen compare
    cmp = sub.add_parser("compare", aliases=["cmp"],
                         help="Compare trained models side-by-side",
                         description="Sort and compare models by parameters or best metric.")
    cmp.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")
    cmp.add_argument("--sort", "-s", default="params",
                     choices=["params", "accuracy", "loss", "recon_loss"],
                     help="Sort by (default: params).")
    cmp.add_argument("--top", "-n", type=int, default=None, help="Show only top N models.")

    # netgen clean
    cln = sub.add_parser("clean", aliases=["rm"],
                         help="Remove models by criteria",
                         description="Delete model folders. Use --dry-run first to preview.")
    cln.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")
    cln.add_argument("--dry-run", action="store_true", default=True,
                     help="Preview only, don't delete (default).")
    cln.add_argument("--force", dest="dry_run", action="store_false",
                     help="Actually delete (use with care).")
    cln.add_argument("--keep-best", "-k", type=int, default=None,
                     help="Keep only the top N best models.")
    cln.add_argument("--status", choices=["generated", "trained"],
                     help="Only remove models with this status.")
    cln.add_argument("--untrained", action="store_true",
                     help="Remove all untrained models.")

    # netgen benchmark
    bm = sub.add_parser("benchmark", aliases=["bm"],
                         help="Train all models and rank by performance",
                         description="Train all untrained models with the same settings, then rank by best metric.")
    bm.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")
    bm.add_argument("--epochs", "-e", type=int, default=10, help="Training epochs per model (default: 10).")
    bm.add_argument("--lr", type=float, default=None, help="Learning rate (uses config default if not set).")
    bm.add_argument("--batch-size", type=int, dest="batch_size", default=None, help="Batch size.")
    bm.add_argument("--seed", type=int, default=42, help="Random seed.")
    bm.add_argument("--device", default=None,
                    help="Device priority override for all models, e.g. 'cuda,mps' (cpu always final fallback).")
    bm.add_argument("--retries", type=int, default=1,
                    help="Automatic retries per failed model (default: 1).")
    bm.add_argument("--workers", "-w", type=int, default=1,
                    help="Train N models concurrently (default: 1).")
    bm.add_argument("--force", action="store_true",
                    help="Re-train already-trained models too.")
    bm.add_argument("--time-budget", type=float, default=None, metavar="MIN",
                    help="Overall wall-clock budget in minutes; split across "
                         "models as per-model timeouts (half-epochs retry).")

    # netgen train
    tr = sub.add_parser("train", aliases=["fit"],
                        help="Train a single model",
                        description="Train one generated model by running its train.py (live output).")
    tr.add_argument("id", help="Model ID (e.g. 001) or folder name.")
    tr.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")
    tr.add_argument("--epochs", "-e", type=int, default=None, help="Training epochs (uses config default).")
    tr.add_argument("--lr", type=float, default=None, help="Learning rate (uses config default).")
    tr.add_argument("--batch-size", type=int, dest="batch_size", default=None, help="Batch size.")
    tr.add_argument("--device", default=None,
                    help="Device priority override, e.g. 'cuda,mps' (cpu always final fallback).")
    tr.add_argument("--seed", type=int, default=None, help="Random seed.")

    # netgen eval
    ev = sub.add_parser("eval", help="Evaluate a single model",
                        description="Run a trained model's eval.py (live output).")
    ev.add_argument("id", help="Model ID (e.g. 001) or folder name.")
    ev.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")

    # netgen sweep
    sw = sub.add_parser("sweep", help="Hyperparameter grid search for one model",
                        description="Tries lr × batch_size combos, re-trains the "
                                    "winner and writes it into config.py.")
    sw.add_argument("id", help="Model ID (e.g. 001) or folder name.")
    sw.add_argument("--epochs", "-e", type=int, default=5, help="Epochs per combo (default: 5).")
    sw.add_argument("--lrs", default=None, help="Comma-separated learning rates, e.g. '0.001,0.01,0.0001'.")
    sw.add_argument("--batches", default=None, help="Comma-separated batch sizes, e.g. '64,128'.")
    sw.add_argument("--device", default=None, help="Device priority override, e.g. 'cuda,mps'.")
    sw.add_argument("--seed", type=int, default=42, help="Random seed.")
    sw.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")

    # netgen export
    exp = sub.add_parser("export", help="Export model comparison report",
                         description="Export model metadata and metrics to a file.")
    exp.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")
    exp.add_argument("--format", "-f", choices=["md", "csv", "json"], default="md",
                     help="Output format (default: md).")
    exp.add_argument("--output", "-o", default=None, help="Output file path.")

    # netgen archs
    arch = sub.add_parser("archs", aliases=["architectures"],
                          help="List and browse all available architectures",
                          description="Display architecture family tree or detailed list.")
    arch.add_argument("--tree", "-t", action="store_true", default=True,
                      help="Show as family tree (default).")
    arch.add_argument("--list", "-l", dest="tree", action="store_false",
                      help="Show as flat list with descriptions.")

    return parser


def run(args: Optional[list[str]] = None) -> int:
    parser = build_parser()

    # Backward compatibility: if first arg looks like --range, treat as generate
    if args is None:
        args = sys.argv[1:]
    if args and not args[0].startswith('-'):
        # Has a subcommand
        pass
    elif args and args[0].startswith('--'):
        # Old style: --range ... → treat as generate
        args = ['generate'] + args

    opts = parser.parse_args(args)

    if opts.command in (None, 'generate', 'gen'):
        if not hasattr(opts, 'range'):
            parser.print_help()
            return 0
        return _cmd_generate(opts)
    elif opts.command in ('list', 'ls'):
        return _cmd_list(opts) or 0
    elif opts.command in ('info', 'inspect', 'show'):
        return _cmd_info(opts) or 0
    elif opts.command in ('compare', 'cmp'):
        return _cmd_compare(opts) or 0
    elif opts.command in ('clean', 'rm'):
        return _cmd_clean(opts) or 0
    elif opts.command in ('benchmark', 'bm'):
        return _cmd_benchmark(opts) or 0
    elif opts.command in ('train', 'fit'):
        return _cmd_train(opts) or 0
    elif opts.command == 'eval':
        return _cmd_eval(opts) or 0
    elif opts.command == 'sweep':
        return _cmd_sweep(opts) or 0
    elif opts.command in ('archs', 'architectures'):
        return _cmd_archs(opts) or 0
    elif opts.command == 'export':
        return _cmd_export(opts) or 0
    else:
        parser.print_help()
        return 0


def main():
    sys.exit(run())
