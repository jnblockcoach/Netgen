"""Model manager — scan, inspect, compare, clean generated model folders."""
import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInfo:
    """Structured info about a generated model folder."""
    index: int
    folder_name: str
    folder_path: str
    architecture: str = ''
    params: int = 0
    dataset: str = 'syn'
    tier: str = 'quick'
    model_type: str = ''
    status: str = 'generated'  # generated | trained | error
    best_metric_name: str = ''
    best_metric_value: Optional[float] = None
    history: list = field(default_factory=list)
    has_best: bool = False
    has_checkpoints: bool = False
    error_msg: str = ''


def scan_models(directory: str) -> list[ModelInfo]:
    """Scan a directory for generated model folders, return parsed info."""
    if not os.path.isdir(directory):
        return []

    models = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isdir(path):
            continue
        # Match pattern: NNN-arch-description
        match = re.match(r'^(\d{3})-(.+)', name)
        if not match:
            continue
        if not os.path.exists(os.path.join(path, 'config.py')):
            continue

        info = _parse_model(path, int(match.group(1)), name)
        models.append(info)

    return models


def _parse_model(path: str, index: int, folder_name: str) -> ModelInfo:
    """Parse a single model folder into ModelInfo."""
    info = ModelInfo(index=index, folder_name=folder_name, folder_path=path)

    # Parse config.py
    config_path = os.path.join(path, 'config.py')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_text = f.read()
        # Extract values
        for key, pattern in [
            ('dataset', r"DATASET\s*=\s*'(\w+)'"),
            ('dataset2', r'DATASET\s*=\s*"(\w+)"'),
        ]:
            m = re.search(pattern, config_text)
            if m:
                info.dataset = m.group(1)
                break
        m = re.search(r'INPUT_DIM\s*=\s*(\d+)', config_text)
        # params from README
        readme_path = os.path.join(path, 'README.md')
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_text = f.read()
            m = re.search(r'\*\*Parameters\*\*:\s*([\d,]+)', readme_text)
            if m:
                info.params = int(m.group(1).replace(',', ''))
            # Architecture from first line: # arch-...
            m = re.search(r'^#\s+(.+)$', readme_text, re.MULTILINE)
            if m:
                info.architecture = m.group(1).strip()
            # Tier
            if 'Production' in readme_text:
                info.tier = 'production'
            elif 'Standard' in readme_text:
                info.tier = 'standard'

    # Determine status
    model_pth = os.path.join(path, 'model.pth')
    best_pth = os.path.join(path, 'best_model.pth')
    log_md = os.path.join(path, 'training_log.md')
    checkpoint_dir = os.path.join(path, 'checkpoints')

    info.has_best = os.path.exists(best_pth)
    info.has_checkpoints = os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir)

    if os.path.exists(model_pth) or info.has_best:
        info.status = 'trained'
    else:
        info.status = 'generated'

    # Parse training_log.md
    if os.path.exists(log_md):
        with open(log_md, 'r', encoding='utf-8') as f:
            log_text = f.read()
        info.history = _parse_log(log_text)
        if info.history:
            # Best metric from log. Validation metrics (added by the
            # val-split training templates) are preferred — they rank
            # generalization instead of train-set overfitting.
            if any('val_loss' in h or 'val_acc' in h for h in info.history):
                if any('val_acc' in h for h in info.history):
                    info.best_metric_name = 'val_acc'
                    info.best_metric_value = max(
                        (h.get('val_acc', h.get('accuracy', 0)) for h in info.history),
                        default=None)
                else:
                    info.best_metric_name = 'val_loss'
                    info.best_metric_value = min(
                        (h.get('val_loss', h.get('loss', float('inf'))) for h in info.history),
                        default=None)
            elif 'Accuracy' in log_text or 'accuracy' in log_text.lower() or 'Acc' in log_text:
                info.best_metric_name = 'accuracy'
                info.best_metric_value = max(
                    (h.get('accuracy', h.get('acc', 0)) for h in info.history),
                    default=None
                )
            elif 'Recon' in log_text:
                info.best_metric_name = 'recon_loss'
                info.best_metric_value = min(
                    (h.get('recon_loss', h.get('loss', float('inf'))) for h in info.history),
                    default=None
                )
            elif 'G Loss' in log_text or 'g_loss' in log_text:
                info.best_metric_name = 'g_loss'
                info.best_metric_value = min(
                    (h.get('g_loss', h.get('loss', float('inf'))) for h in info.history),
                    default=None
                )
            else:
                info.best_metric_name = 'loss'
                info.best_metric_value = min(
                    (h.get('loss', float('inf'))) for h in info.history
                )

    return info


def _parse_log(text: str) -> list[dict]:
    """Parse training_log.md table into list of dicts."""
    history = []
    # Find table rows: | epoch | value1 | [value2] | [value3] |
    lines = text.split('\n')
    in_table = False
    headers = []
    for line in lines:
        line = line.strip()
        if line.startswith('| Epoch') or line.startswith('|-------'):
            in_table = True
            if line.startswith('| Epoch'):
                headers = [h.strip().lower().replace(' ', '_') for h in line.split('|')[1:-1]]
            continue
        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 2:
                try:
                    epoch = int(parts[0])
                    entry = {'epoch': epoch}
                    for i, h in enumerate(headers[1:], 1):
                        if i < len(parts) and parts[i]:
                            try:
                                entry[h] = float(parts[i])
                            except ValueError:
                                pass
                    history.append(entry)
                except ValueError:
                    continue
        elif in_table and not line.startswith('|'):
            in_table = False
    return history


def list_models(directory: str) -> str:
    """Generate a formatted table of all models."""
    models = scan_models(directory)
    if not models:
        return f"No models found in {directory}"

    trained = sum(1 for m in models if m.status == 'trained')
    generated = sum(1 for m in models if m.status == 'generated')

    lines = [
        f"Models in: {directory} ({len(models)} total, {trained} trained, {generated} pending)",
        "",
        f"{'ID':>5s}  {'Architecture':<25s}  {'Params':>12s}  {'Dataset':<8s}  {'Status':<10s}  {'Best Metric':<18s}",
        f"{'-'*5}  {'-'*25}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*18}",
    ]
    for m in models:
        metric_str = ''
        if m.best_metric_value is not None:
            metric_str = f'{m.best_metric_name}={m.best_metric_value:.4f}'
        lines.append(
            f"{m.index:5d}  {m.architecture:<25s}  {m.params:>12,}  {m.dataset:<8s}  {m.status:<10s}  {metric_str:<18s}"
        )
    return '\n'.join(lines)


def _find_model(models: list, identifier: str) -> Optional[ModelInfo]:
    """Locate a model by ID, exact folder name, or substring."""
    for m in models:
        if str(m.index) == identifier or m.folder_name == identifier or identifier in m.folder_name:
            return m
    return None


def info_model(directory: str, identifier: str) -> str:
    """Show detailed info for a specific model."""
    models = scan_models(directory)
    model = _find_model(models, identifier)
    if model is None:
        return f"Model not found: {identifier}"

    lines = [
        f"",
        f"  {model.folder_name}",
        f"  {'='*40}",
        f"",
        f"  Architecture:  {model.architecture}",
        f"  Parameters:    {model.params:,}",
        f"  Dataset:       {model.dataset}",
        f"  Tier:          {model.tier}",
        f"  Status:        {model.status}",
        f"  Best model:    {'[x]' if model.has_best else '[ ]'}",
        f"  Checkpoints:   {'[x]' if model.has_checkpoints else '[ ]'}",
    ]

    if model.best_metric_value is not None:
        lines.append(f"  Best {model.best_metric_name}: {model.best_metric_value:.4f}")

    lines.append(f"")
    lines.append(f"  Files:")
    for f in sorted(os.listdir(model.folder_path)):
        fpath = os.path.join(model.folder_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            lines.append(f"    {f:<25s} {size:>10,} B")
        elif os.path.isdir(fpath) and f != '__pycache__':
            count = len(os.listdir(fpath))
            lines.append(f"    {f}/ {'(' + str(count) + ' files)':<20s}")

    if model.history:
        lines.append(f"")
        lines.append(f"  Training History ({len(model.history)} epochs):")
        lines.append(f"  {'Epoch':>6s}  {'Loss':>10s}" +
                     (f"  {'Acc':>10s}" if 'accuracy' in model.history[0] or 'acc' in model.history[0] else ""))
        for h in model.history[:10]:
            acc_str = ''
            acc_val = h.get('accuracy', h.get('acc'))
            if acc_val is not None:
                acc_str = f"  {acc_val:10.4f}"
            lines.append(f"  {h['epoch']:6d}  {h.get('loss', 0):10.4f}{acc_str}")
        if len(model.history) > 10:
            lines.append(f"  ... ({len(model.history) - 10} more epochs)")

    return '\n'.join(lines)


def compare_models(directory: str, sort_by: str = 'params', top_n: int = None,
                   reverse: bool = False) -> str:
    """Compare all models, sorted by a metric."""
    models = scan_models(directory)
    if not models:
        return f"No models found in {directory}"

    # Only trained models for metric-based sorting
    if sort_by in ('accuracy', 'acc', 'loss', 'recon_loss'):
        models = [m for m in models if m.best_metric_value is not None]
        if sort_by in ('accuracy', 'acc'):
            models.sort(key=lambda m: m.best_metric_value or 0, reverse=True)
        else:
            models.sort(key=lambda m: m.best_metric_value or float('inf'))
    elif sort_by == 'params':
        models.sort(key=lambda m: m.params, reverse=reverse)
    else:
        models.sort(key=lambda m: m.index)

    if top_n:
        models = models[:top_n]

    lines = [
        f"Comparison ({len(models)} models, sorted by {sort_by}):",
        "",
        f"{'Rank':>5s}  {'ID':>5s}  {'Architecture':<25s}  {'Params':>12s}  {'Tier':<10s}  {'Best Metric':<20s}",
        f"{'-'*5}  {'-'*5}  {'-'*25}  {'-'*12}  {'-'*10}  {'-'*20}",
    ]
    for rank, m in enumerate(models, 1):
        metric_str = ''
        if m.best_metric_value is not None:
            metric_str = f'{m.best_metric_name}={m.best_metric_value:.4f}'
        lines.append(
            f"{rank:5d}  {m.index:5d}  {m.architecture:<25s}  {m.params:>12,}  {m.tier:<10s}  {metric_str:<20s}"
        )
    return '\n'.join(lines)


def clean_models(directory: str, keep_best: int = None, status_filter: str = None,
                 untrained: bool = False, dry_run: bool = True) -> str:
    """Remove models matching criteria."""
    models = scan_models(directory)

    to_remove = []
    for m in models:
        if status_filter and m.status != status_filter:
            continue
        if untrained and m.status == 'trained':
            continue
        if untrained and m.status != 'generated':
            continue
        to_remove.append(m)

    # If keep_best, sort and keep top N
    if keep_best and len(models) - len(to_remove) < keep_best:
        trained = [m for m in models if m.status == 'trained' and m.best_metric_value is not None]
        trained.sort(key=lambda m: m.best_metric_value or 0, reverse=True)
        keep_ids = {m.index for m in trained[:keep_best]}
        to_remove = [m for m in to_remove if m.index not in keep_ids]

    if not to_remove:
        return "Nothing to remove."

    lines = []
    if dry_run:
        lines.append(f"[DRY RUN] Would remove {len(to_remove)} models:")
    else:
        lines.append(f"Removing {len(to_remove)} models:")

    for m in to_remove:
        lines.append(f"  {m.folder_name}  ({m.status}, {m.params:,} params)")
        if not dry_run:
            import shutil
            shutil.rmtree(m.folder_path)

    if dry_run:
        lines.append(f"\nRun without --dry-run to actually delete.")
    else:
        lines.append(f"\nRemoved {len(to_remove)} models.")

    return '\n'.join(lines)


def export_models(directory: str, fmt: str = 'md', output: str = None) -> str:
    """Export model comparison to a file."""
    models = scan_models(directory)
    if not models:
        return "No models to export."

    if output is None:
        output = f'comparison_report.{fmt}'

    if fmt == 'csv':
        lines = ['index,folder,architecture,params,dataset,tier,status,best_metric_name,best_metric_value']
        for m in models:
            lines.append(f'{m.index},{m.folder_name},{m.architecture},{m.params},{m.dataset},{m.tier},{m.status},{m.best_metric_name},{m.best_metric_value}')
        content = '\n'.join(lines)

    elif fmt == 'json':
        data = []
        for m in models:
            data.append({
                'index': m.index,
                'folder': m.folder_name,
                'architecture': m.architecture,
                'params': m.params,
                'dataset': m.dataset,
                'tier': m.tier,
                'status': m.status,
                'best_metric_name': m.best_metric_name,
                'best_metric_value': m.best_metric_value,
                'has_best': m.has_best,
                'has_checkpoints': m.has_checkpoints,
                'history': m.history,
            })
        content = json.dumps(data, indent=2)

    else:  # md
        lines = [
            '# NetGen Model Comparison',
            '',
            f'**Directory**: {directory}',
            f'**Total**: {len(models)} models',
            '',
            '| ID | Architecture | Params | Dataset | Tier | Status | Best Metric |',
            '|----|-------------|--------|---------|------|--------|-------------|',
        ]
        for m in models:
            metric_str = ''
            if m.best_metric_value is not None:
                metric_str = f'{m.best_metric_name}={m.best_metric_value:.4f}'
            lines.append(f'| {m.index:03d} | {m.architecture} | {m.params:,} | {m.dataset} | {m.tier} | {m.status} | {metric_str} |')
        content = '\n'.join(lines)

    with open(output, 'w', encoding='utf-8') as f:
        f.write(content)

    return f"Exported to {output} ({fmt})"


def train_model(directory: str, identifier: str, epochs: int = None,
                lr: float = None, batch_size: int = None, device: str = None,
                seed: int = None) -> str:
    """Train a single model by running its train.py in a subprocess.

    The subprocess inherits stdout/stderr so progress bars show live.
    `device` overrides the model's DEVICE_PRIORITY (e.g. 'cuda,mps').
    """
    import subprocess
    import sys

    model = _find_model(scan_models(directory), identifier)
    if model is None:
        return f"Model not found: {identifier}"
    train_py = os.path.join(model.folder_path, 'train.py')
    if not os.path.exists(train_py):
        return f"No train.py in {model.folder_name}"

    cmd = [sys.executable, 'train.py']
    if epochs is not None:
        cmd += ['--epochs', str(epochs)]
    if lr is not None:
        cmd += ['--lr', str(lr)]
    if batch_size is not None:
        cmd += ['--batch-size', str(batch_size)]
    if device:
        cmd += ['--device', device]
    if seed is not None:
        cmd += ['--seed', str(seed)]

    print(f"\n  Training {model.folder_name} ...\n")
    try:
        result = subprocess.run(cmd, cwd=model.folder_path, timeout=7200)
    except subprocess.TimeoutExpired:
        return f"Training {model.folder_name} TIMEOUT (>2h)"
    if result.returncode != 0:
        return f"Training {model.folder_name} FAILED (exit {result.returncode})"
    return f"\n  {model.folder_name}: training OK"


def eval_model(directory: str, identifier: str) -> str:
    """Evaluate a single model by running its eval.py in a subprocess."""
    import subprocess
    import sys

    model = _find_model(scan_models(directory), identifier)
    if model is None:
        return f"Model not found: {identifier}"
    eval_py = os.path.join(model.folder_path, 'eval.py')
    if not os.path.exists(eval_py):
        return f"No eval.py in {model.folder_name}"
    if not os.path.exists(os.path.join(model.folder_path, 'model.pth')):
        return f"{model.folder_name} not trained yet (no model.pth). Run: netgen train {model.index}"

    print(f"\n  Evaluating {model.folder_name} ...\n")
    result = subprocess.run([sys.executable, 'eval.py'], cwd=model.folder_path,
                            timeout=1800)
    if result.returncode != 0:
        return f"Evaluation {model.folder_name} FAILED (exit {result.returncode})"
    return f"\n  {model.folder_name}: evaluation OK"


def benchmark_models(directory: str, epochs: int = 10, lr: float = None,
                    batch_size: int = None, seed: int = 42,
                    device: str = None, retries: int = 1) -> str:
    """Train all untrained models with the same settings, then rank them.

    Args:
        directory: Path to generated models.
        epochs: Training epochs per model (default 10).
        lr: Learning rate (uses model's config default if None).
        batch_size: Batch size (uses model's config default if None).
        seed: Random seed.

    Returns:
        Formatted leaderboard string.
    """
    import subprocess
    import sys
    import time

    models = scan_models(directory)
    untrained = [m for m in models if m.status == 'generated']

    if not untrained:
        return "All models already trained. Use 'netgen list' to see status."

    results = []
    n = len(untrained)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {n} models × {epochs} epochs")
    print(f"{'='*60}\n")

    for i, model in enumerate(untrained, 1):
        train_py = os.path.join(model.folder_path, 'train.py')
        if not os.path.exists(train_py):
            print(f"  [{i}/{n}] {model.folder_name}  SKIP (no train.py)")
            continue

        # Build command
        cmd = [sys.executable, 'train.py', '--epochs', str(epochs), '--seed', str(seed)]
        if lr is not None:
            cmd += ['--lr', str(lr)]
        if batch_size is not None:
            cmd += ['--batch-size', str(batch_size)]
        if device:
            cmd += ['--device', device]

        t0 = time.time()
        print(f"  [{i}/{n}] {model.folder_name:<35s} ", end='', flush=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                    cwd=model.folder_path)
            elapsed = time.time() - t0

            if result.returncode != 0:
                # One automatic retry (transient failures happen, e.g. OOM at init)
                if retries > 0:
                    print(f"FAIL ({elapsed:.1f}s) -> retry... ", end='', flush=True)
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                            timeout=600, cwd=model.folder_path)
                    elapsed = time.time() - t0
                if result.returncode != 0:
                    print(f"FAIL ({elapsed:.1f}s)")
                    model.status = 'error'
                    model.error_msg = result.stderr[-200:] if result.stderr else 'Unknown error'
                    continue

            # Re-parse the model to get updated metrics
            updated = _parse_model(model.folder_path, model.index, model.folder_name)
            metric_val = updated.best_metric_value or 0
            metric_name = updated.best_metric_name or 'loss'

            results.append({
                'model': updated,
                'elapsed': elapsed,
                'metric_val': metric_val,
                'metric_name': metric_name,
            })
            print(f"OK ({elapsed:.1f}s)  {metric_name}={metric_val:.4f}")

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            model.status = 'error'
            model.error_msg = 'Timeout (>10min)'
        except Exception as e:
            print(f"ERROR: {e}")

    # ── Leaderboard ──
    if not results:
        return "\nNo models completed training."

    # Sort: accuracy/acc higher=better, loss/recon_loss/g_loss lower=better
    def _sort_key(r):
        nm = r['metric_name']
        val = r['metric_val']
        # Loss metrics: lower is better → negate for ascending sort
        if nm in ('loss', 'recon_loss', 'g_loss', 'd_loss'):
            return val
        # Accuracy metrics: higher is better → negate for descending sort
        return -val
    results.sort(key=_sort_key)

    lines = [
        "",
        f"{'='*70}",
        f"  BENCHMARK RESULTS — {len(results)} models × {epochs} epochs",
        f"{'='*70}",
        "",
        f"{'Rank':>4s}  {'Model':<30s}  {'Params':>10s}  {'Metric':<20s}  {'Time':>8s}",
        f"{'-'*4}  {'-'*30}  {'-'*10}  {'-'*20}  {'-'*8}",
    ]
    for rank, r in enumerate(results, 1):
        m = r['model']
        metric_str = f"{r['metric_name']}={r['metric_val']:.4f}"
        lines.append(
            f"{rank:4d}  {m.folder_name:<30s}  {m.params:>10,}  {metric_str:<20s}  {r['elapsed']:>7.1f}s"
        )

    # Also write report
    report_path = os.path.join(directory, 'benchmark_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Report\n\n")
        f.write(f"**Epochs**: {epochs}  \n")
        f.write(f"**Models**: {len(results)} trained  \n\n")
        f.write(f"| Rank | Model | Params | Metric | Time |\n")
        f.write(f"|------|-------|--------|--------|------|\n")
        for rank, r in enumerate(results, 1):
            m = r['model']
            metric_str = f"{r['metric_name']}={r['metric_val']:.4f}"
            f.write(f"| {rank} | {m.folder_name} | {m.params:,} | {metric_str} | {r['elapsed']:.1f}s |\n")
        f.write(f"\n*Generated by `netgen benchmark`*\n")

    # ── Loss curve chart (best effort; requires matplotlib) ──
    curve_path = _plot_benchmark_curves(results, directory)
    if curve_path:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(f"\n## Loss Curves\n\n![benchmark curves]({os.path.basename(curve_path)})\n")

    lines.append(f"\nSaved benchmark_report.md")
    return '\n'.join(lines)


def _plot_benchmark_curves(results: list, directory: str):
    """Plot every trained model's (val) loss curve into benchmark_curves.png.

    Returns the image path, or None if matplotlib is unavailable or there
    is nothing to plot.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for r in results:
        hist = r['model'].history
        if not hist:
            continue
        epochs = [h.get('epoch', i) for i, h in enumerate(hist)]
        if any('val_loss' in h for h in hist):
            key = 'val_loss'
        elif any('loss' in h for h in hist):
            key = 'loss'
        else:
            key = 'recon_loss'
        vals = [h.get(key, float('nan')) for h in hist]
        ax.plot(epochs, vals, label=r['model'].folder_name)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (val when available)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title('Benchmark — Loss Curves')
    plt.tight_layout()
    out = os.path.join(directory, 'benchmark_curves.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out
