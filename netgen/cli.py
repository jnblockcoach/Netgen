"""Command-line interface for NetGen."""
import argparse
import os
import sys
from typing import Optional

from .search import find_candidates, list_architectures
from .generator import gen_folder


# Model type → short label mapping
_TYPE_LABELS = {
    'ce': 'cls', 'mse': 'reg', 'rnn': 'seq',
    'cnn': 'img', 'ae': 'ae', 'vae': 'vae',
    'mt': 'mt', 'gan': 'gan', 'contrastive': 'ctr',
    'siamese': 'sim',
}

# Which architectures are compatible with each dataset task
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

# Architectures compatible with each task type
# Non-sequential classification: no RNNs (data is flat vectors, not sequences)
_TASK_ARCHS = {
    'classification': ['linear', 'mlp', 'deep', 'wide', 'resblock', 'highway', 'moe', 'cnn', 'multitask'],
    'regression':    ['linear', 'mlp', 'deep', 'wide', 'resblock'],
}


def parse_range(range_str: str) -> tuple[int, int]:
    """Parse 'lo-hi' range string into (low, high) ints."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="netgen",
        description="NetGen - Batch Neural Network Model Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Examples:
  netgen --range 10000-20000 --count 20
  netgen --range 10-100 --count 5 --dataset iris
  netgen --range 100000-1000000 --count 10 --dataset mnist
  netgen --range 5000000000-10000000000 --count 3
  netgen --range 10000-20000 --count 20 --arch mlp,lstm,cnn

Available architectures:
  {', '.join(list_architectures())}

Available datasets:
  syn, iris, wine, breast_cancer, moons, circles, blobs, mnist, cifar10, text, line
        """)
    parser.add_argument("--range", required=True,
                        help="Parameter count range, e.g. '10000-20000'.")
    parser.add_argument("--count", type=int, default=20,
                        help="Number of models to generate (default: 20).")
    parser.add_argument("--output", "-o", default="./generated_models",
                        help="Output directory (default: ./generated_models).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--dataset", default="syn",
                        help="Dataset name (default: syn).")
    parser.add_argument("--arch", default=None,
                        help="Comma-separated architecture filter, e.g. 'mlp,cnn,lstm'.")
    return parser


def run(args: Optional[list[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        args: Command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = build_parser()
    opts = parser.parse_args(args)

    # Parse parameter range
    try:
        lo, hi = parse_range(opts.range)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Parse architecture filter
    arch_filter = None
    if opts.arch:
        arch_filter = [a.strip() for a in opts.arch.split(",") if a.strip()]
        valid = list_architectures()
        invalid = [a for a in arch_filter if a not in valid]
        if invalid:
            print(f"Warning: Unknown architectures ignored: {invalid}", file=sys.stderr)
            print(f"  Available: {valid}", file=sys.stderr)
            arch_filter = [a for a in arch_filter if a in valid]
        if not arch_filter:
            print("Error: No valid architectures after filtering.", file=sys.stderr)
            return 1

    # When using a real dataset, auto-filter to compatible architectures
    if opts.dataset != 'syn' and opts.dataset in _DATASET_TASK:
        task = _DATASET_TASK[opts.dataset]
        compat = _TASK_ARCHS.get(task, [])
        if arch_filter:
            # User already specified — warn if incompatible
            bad = [a for a in arch_filter if a not in compat]
            if bad:
                print(f"Warning: {bad} not designed for {task}. Consider: {compat}", file=sys.stderr)
        else:
            arch_filter = compat

    output_dir = os.path.abspath(opts.output)

    print(f"\nNetGen - Generating {opts.count} models in range {lo:,}-{hi:,}")
    if arch_filter:
        print(f"Architectures: {', '.join(arch_filter)}")
    print(f"Dataset: {opts.dataset}")
    print(f"Output: {output_dir}\n")

    # Find candidates
    candidates = find_candidates(lo, hi, opts.count, opts.seed, arch_filter)
    candidates.sort(key=lambda x: x[2])  # sort by param count

    if len(candidates) == 0:
        print(f"Warning: Could not find any architectures in range {lo:,}-{hi:,}", file=sys.stderr)
        print("Try widening the range or using --arch to select different architectures.", file=sys.stderr)
        return 1

    if len(candidates) < opts.count:
        print(f"Note: Found {len(candidates)} candidates (requested {opts.count}).")

    # Generate folders
    generated = 0
    for i, (description, code, params, input_dim, output_dim, model_type) in enumerate(
            candidates[:opts.count]):
        folder = gen_folder(
            output_dir, i + 1, description, code, "M{}",
            params, input_dim, output_dim, model_type, opts.dataset
        )
        label = _TYPE_LABELS.get(model_type, model_type)
        print(f"  [{i + 1:3d}] {os.path.basename(folder)}: {params:>12,} params  [{label}]")
        generated += 1

    print(f"\nGenerated {generated} models in {output_dir}")
    print("Each folder: model.py, data.py, train.py, eval.py, predict.py,")
    print("           visualize.py, config.py, requirements.txt, README.md")
    print("To train: cd <folder> && python train.py --lr 0.01 --epochs 50 --batch-size 128")
    return 0


def main():
    """Entry point for console_scripts."""
    sys.exit(run())
