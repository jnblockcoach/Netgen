"""Command-line interface for NetGen — generate + manage models."""
import argparse
import os
import sys
from typing import Optional

from .search import find_candidates, list_architectures
from .generator import gen_folder
from .manager import (
    list_models, info_model, compare_models, clean_models, export_models
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

_TASK_ARCHS = {
    'classification': ['linear', 'mlp', 'deep', 'wide', 'resblock', 'highway', 'moe', 'cnn', 'multitask'],
    'regression':    ['linear', 'mlp', 'deep', 'wide', 'resblock'],
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


def _resolve_arch_filter(opts, dataset: str) -> Optional[list[str]]:
    """Resolve architecture filter from --arch and dataset compatibility."""
    arch_filter = None
    if opts.arch:
        arch_filter = [a.strip() for a in opts.arch.split(",") if a.strip()]
        valid = list_architectures()
        invalid = [a for a in arch_filter if a not in valid]
        if invalid:
            print(f"Warning: Unknown architectures ignored: {invalid}", file=sys.stderr)
            arch_filter = [a for a in arch_filter if a in valid]
        if not arch_filter:
            print("Error: No valid architectures after filtering.", file=sys.stderr)
            return None

    if dataset != 'syn' and dataset in _DATASET_TASK:
        task = _DATASET_TASK[dataset]
        compat = _TASK_ARCHS.get(task, [])
        if arch_filter:
            bad = [a for a in arch_filter if a not in compat]
            if bad:
                print(f"Warning: {bad} not designed for {task}. Compatible: {compat}", file=sys.stderr)
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

    output_dir = os.path.abspath(args.output)

    print(f"\nNetGen - Generating {args.count} models in range {lo:,}-{hi:,}")
    if arch_filter:
        print(f"Architectures: {', '.join(arch_filter)}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}\n")

    candidates = find_candidates(lo, hi, args.count, args.seed, arch_filter)
    candidates.sort(key=lambda x: x[2])

    if len(candidates) == 0:
        print(f"Warning: Could not find any architectures in range {lo:,}-{hi:,}", file=sys.stderr)
        print("Try widening the range or using --arch to select different architectures.", file=sys.stderr)
        return 1

    if len(candidates) < args.count:
        print(f"Note: Found {len(candidates)} candidates (requested {args.count}).")

    generated = 0
    for i, (description, code, params, input_dim, output_dim, model_type) in enumerate(
            candidates[:args.count]):
        folder = gen_folder(
            output_dir, i + 1, description, code, "M{}",
            params, input_dim, output_dim, model_type, args.dataset
        )
        label = _TYPE_LABELS.get(model_type, model_type)
        print(f"  [{i + 1:3d}] {os.path.basename(folder)}: {params:>12,} params  [{label}]")
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


# ── Subcommand: export ──

def _cmd_export(args):
    print(export_models(args.dir, args.format, args.output))


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
    gen.add_argument("--dataset", default="syn", help="Dataset name (default: syn).")
    gen.add_argument("--arch", default=None, help="Architecture filter, e.g. 'mlp,cnn,lstm'.")

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

    # netgen export
    exp = sub.add_parser("export", help="Export model comparison report",
                         description="Export model metadata and metrics to a file.")
    exp.add_argument("--dir", "-d", default="./generated_models", help="Models directory.")
    exp.add_argument("--format", "-f", choices=["md", "csv", "json"], default="md",
                     help="Output format (default: md).")
    exp.add_argument("--output", "-o", default=None, help="Output file path.")

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
    elif opts.command == 'export':
        return _cmd_export(opts) or 0
    else:
        parser.print_help()
        return 0


def main():
    sys.exit(run())
