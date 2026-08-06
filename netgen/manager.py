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
            def _clean(vals):
                """Drop NaN values (models without val metrics log NaN)."""
                return [v for v in vals if v == v]

            if any('val_loss' in h or 'val_acc' in h for h in info.history):
                if any('val_acc' in h for h in info.history):
                    vals = _clean(h.get('val_acc', h.get('accuracy', 0))
                                  for h in info.history)
                    if vals:
                        info.best_metric_name = 'val_acc'
                        info.best_metric_value = max(vals)
                if info.best_metric_name != 'val_acc':
                    vals = _clean(h.get('val_loss', h.get('loss', float('inf')))
                                  for h in info.history)
                    if vals:
                        info.best_metric_name = 'val_loss'
                        info.best_metric_value = min(vals)
            if not info.best_metric_name and ('Accuracy' in log_text
                                              or 'accuracy' in log_text.lower()
                                              or 'Acc' in log_text):
                info.best_metric_name = 'accuracy'
                info.best_metric_value = max(
                    _clean(h.get('accuracy', h.get('acc', 0)) for h in info.history),
                    default=None
                )
            elif not info.best_metric_name and 'Recon' in log_text:
                info.best_metric_name = 'recon_loss'
                info.best_metric_value = min(
                    _clean(h.get('recon_loss', h.get('loss', float('inf'))) for h in info.history),
                    default=None
                )
            elif not info.best_metric_name and ('G Loss' in log_text or 'g_loss' in log_text):
                info.best_metric_name = 'g_loss'
                info.best_metric_value = min(
                    _clean(h.get('g_loss', h.get('loss', float('inf'))) for h in info.history),
                    default=None
                )
            elif not info.best_metric_name:
                info.best_metric_name = 'loss'
                info.best_metric_value = min(
                    _clean(h.get('loss', float('inf')) for h in info.history),
                    default=None
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
                                val = float(parts[i])
                                if val == val:  # drop NaN (no val metric logged)
                                    entry[h] = val
                            except ValueError:
                                pass
                    history.append(entry)
                except ValueError:
                    continue
        elif in_table and not line.startswith('|'):
            in_table = False
    return history


def read_config(config_path: str) -> dict:
    """Parse simple `KEY = value` lines from a config.py into a dict.

    Strings are unquoted, numbers converted, comments ignored.
    """
    values = {}
    try:
        lines = open(config_path, encoding='utf-8').read().splitlines()
    except OSError:
        return values
    for line in lines:
        m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$', line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if (raw.startswith("'") and raw.endswith("'")) or \
           (raw.startswith('"') and raw.endswith('"')):
            values[key] = raw[1:-1]
        elif raw.startswith('[') and raw.endswith(']'):  # list like ['cuda','mps']
            values[key] = [s.strip().strip("'\"").strip('\"')
                           for s in raw[1:-1].split(',') if s.strip()]
        else:
            try:
                values[key] = int(raw)
            except ValueError:
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = raw
    return values


def update_config(config_path: str, **overrides) -> dict:
    """Rewrite `KEY = value` lines in config.py, preserving comments.

    Returns the dict of keys that were actually changed.
    """
    if not overrides:
        return {}
    try:
        lines = open(config_path, encoding='utf-8').read().splitlines(keepends=True)
    except OSError:
        return {}
    changed = {}
    for i, line in enumerate(lines):
        m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=', line)
        if not m or m.group(1) not in overrides:
            continue
        key = m.group(1)
        new_val = overrides[key]
        comment = ''
        cm = re.search(r'(\s*#.*)$', line)
        if cm:
            comment = cm.group(1)
        if isinstance(new_val, bool):
            rendered = 'True' if new_val else 'False'
        elif isinstance(new_val, str):
            rendered = f"'{new_val}'"
        elif isinstance(new_val, (list, tuple)):
            rendered = "[" + ', '.join(f"'{v}'" for v in new_val) + "]"
        else:
            rendered = repr(new_val)
        lines[i] = f"{key} = {rendered}{comment}\n"
        changed[key] = new_val
    if changed:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    return changed


def _coerce_config_value(key: str, v):
    """Coerce CLI-supplied strings to proper config.py types.

    '50' -> 50, '0.01' -> 0.01, 'adamw' -> 'adamw',
    'cuda,mps' -> ['cuda', 'mps'] for DEVICE_PRIORITY.
    """
    if not isinstance(v, str):
        return v
    if key == 'DEVICE_PRIORITY':
        return [s.strip() for s in v.split(',') if s.strip()]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def set_model_params(directory: str, identifier: str,
                     **overrides) -> str:
    """Adjust a model's training hyperparameters in its config.py.

    Usage (TUI command bar / CLI):
        set_model_params('.', '001', EPOCHS=50, LR=0.01, OPTIMIZER='adamw')
    """
    overrides = {k: _coerce_config_value(k, v) for k, v in overrides.items()}
    model = _find_model(scan_models(directory), identifier)
    if model is None:
        return f"Model not found: {identifier}"
    config_path = os.path.join(model.folder_path, 'config.py')
    if not os.path.exists(config_path):
        return f"No config.py in {model.folder_name}"
    changed = update_config(config_path, **overrides)
    if not changed:
        return f"No recognized keys changed in {model.folder_name}"
    current = read_config(config_path)
    summary = ", ".join(f"{k}={current.get(k, v)}"
                        for k, v in changed.items())
    return f"{model.folder_name}: updated {summary}"


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

    # Only trained models for metric-based sorting.
    # Higher-is-better: accuracy, acc, val_acc. Lower-is-better: the rest.
    if sort_by in ('accuracy', 'acc', 'loss', 'recon_loss',
                   'val_acc', 'val_loss', 'g_loss', 'd_loss'):
        models = [m for m in models if m.best_metric_value is not None]
        if sort_by in ('accuracy', 'acc', 'val_acc'):
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


def sweep_model(directory: str, identifier: str, epochs: int = 5,
                lrs: list = None, batches: list = None, device: str = None,
                seed: int = 42) -> str:
    """Grid-search hyperparameters (lr × batch_size) for a single model.

    Each combo is trained with the model's own train.py; the winner is
    re-trained (final model.pth) and its hyperparameters are written back
    into config.py so later plain `netgen train <id>` uses them.

    Returns:
        Formatted sweep report (also saved to sweep_report.md).
    """
    import subprocess
    import sys
    import time
    import itertools

    model = _find_model(scan_models(directory), identifier)
    if model is None:
        return f"Model not found: {identifier}"
    train_py = os.path.join(model.folder_path, 'train.py')
    if not os.path.exists(train_py):
        return f"No train.py in {model.folder_name}"

    lrs = lrs or [0.001, 0.01]
    batches = batches or [64, 128]
    combos = list(itertools.product(lrs, batches))

    print(f"\n  Sweep {model.folder_name}: {len(combos)} combos × {epochs} epochs"
          f" (lr={lrs}, batch={batches})\n")

    results = []
    for lr, bs in combos:
        cmd = [sys.executable, 'train.py', '--epochs', str(epochs),
               '--lr', str(lr), '--batch-size', str(bs), '--seed', str(seed)]
        if device:
            cmd += ['--device', device]
        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=1800, cwd=model.folder_path)
            elapsed = time.time() - t0
            if result.returncode == 0:
                updated = _parse_model(model.folder_path, model.index, model.folder_name)
                metric = (updated.best_metric_name or 'loss', updated.best_metric_value or 0)
                print(f"    lr={lr:<8g} bs={bs:<4d} -> {metric[0]}={metric[1]:.4f} ({elapsed:.1f}s)")
                results.append({'lr': lr, 'bs': bs, 'metric': metric, 'elapsed': elapsed, 'ok': True})
            else:
                print(f"    lr={lr:<8g} bs={bs:<4d} -> FAIL (exit {result.returncode})")
                results.append({'lr': lr, 'bs': bs, 'metric': ('loss', float('inf')),
                                'elapsed': time.time() - t0, 'ok': False})
        except subprocess.TimeoutExpired:
            print(f"    lr={lr:<8g} bs={bs:<4d} -> TIMEOUT (>30min)")
            results.append({'lr': lr, 'bs': bs, 'metric': ('loss', float('inf')),
                            'elapsed': time.time() - t0, 'ok': False})

    ok = [r for r in results if r['ok']]
    if not ok:
        return "\nAll sweep combos failed. Check the model's train.py manually."

    def _key(r):
        name, val = r['metric']
        return val if name in ('loss', 'recon_loss', 'g_loss', 'd_loss', 'val_loss') else -val
    ok.sort(key=_key)
    best = ok[0]

    # ── Re-train with the winning combo (final artifacts) ──
    print(f"\n  Best: lr={best['lr']:g}, bs={best['bs']} "
          f"({best['metric'][0]}={best['metric'][1]:.4f}) -> final run...")
    cmd = [sys.executable, 'train.py', '--epochs', str(epochs),
           '--lr', str(best['lr']), '--batch-size', str(best['bs']),
           '--seed', str(seed)]
    if device:
        cmd += ['--device', device]
    subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                   cwd=model.folder_path)

    # ── Write the winning hyperparameters back into config.py ──
    config_path = os.path.join(model.folder_path, 'config.py')
    try:
        cfg = open(config_path, encoding='utf-8').read()
        import re
        cfg = re.sub(r'^LR = .*', f'LR = {best["lr"]:g}                   # learning rate (set by netgen sweep)',
                     cfg, count=1, flags=re.M)
        cfg = re.sub(r'^BATCH_SIZE = .*',
                     f'BATCH_SIZE = {best["bs"]}              # batch size (set by netgen sweep)',
                     cfg, count=1, flags=re.M)
        open(config_path, 'w', encoding='utf-8').write(cfg)
    except OSError:
        pass

    # ── Report ──
    lines = [
        "",
        f"  SWEEP RESULTS — {model.folder_name}",
        f"  {'lr':<10s} {'batch':<7s} {'metric':<20s} {'time':>7s}",
        f"  {'-'*10} {'-'*7} {'-'*20} {'-'*7}",
    ]
    for r in ok:
        marker = " *" if r is best else ""
        lines.append(
            f"  {r['lr']:<10g} {r['bs']:<7d} {r['metric'][0]}={r['metric'][1]:.4f}{marker:<10s} {r['elapsed']:>6.1f}s")
    lines.append(f"\n  Best: lr={best['lr']:g}, batch={best['bs']} — written to config.py")
    report_path = os.path.join(directory, 'sweep_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Sweep Report — {model.folder_name}\n\n")
        f.write(f"**Epochs**: {epochs}  \n")
        f.write(f"| lr | batch | metric | time |\n")
        f.write(f"|----|-------|--------|------|\n")
        for r in ok:
            star = " **" if r is best else ""
            f.write(f"| {r['lr']:g} | {r['bs']} | {r['metric'][0]}={r['metric'][1]:.4f}{star} | {r['elapsed']:.1f}s |\n")
        f.write(f"\n*Generated by `netgen sweep`*\n")
    lines.append(f"\nSaved sweep_report.md")
    return '\n'.join(lines)


def benchmark_models(directory: str, epochs: int = 10, lr: float = None,
                    batch_size: int = None, seed: int = 42,
                    device: str = None, retries: int = 1, workers: int = 1,
                    force: bool = False, time_budget: float = None) -> str:
    """Train models with the same settings, then rank them.

    Args:
        directory: Path to generated models.
        epochs: Training epochs per model (default 10).
        lr: Learning rate (uses model's config default if None).
        batch_size: Batch size (uses model's config default if None).
        seed: Random seed.
        device: Device priority override (e.g. 'cuda,mps').
        retries: Automatic retries per failed model (default 1).
        workers: How many models to train concurrently (default 1).
        force: Re-train already-trained models too (default False).
        time_budget: Overall wall-clock budget in minutes. The budget is
            split evenly across models as a per-model timeout; models that
            exceed it retry once with half the epochs.

    Returns:
        Formatted leaderboard string.
    """
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    models = scan_models(directory)
    if not models:
        return f"No models found in {directory}"
    todo = list(models) if force else [m for m in models if m.status == 'generated']

    if not todo:
        return ("All models already trained. Use 'netgen list' to see status, "
                "or --force to re-train everything.")

    if time_budget:
        per_model_budget = time_budget * 60 / max(1, len(todo))
        budget_note = f" | budget {time_budget:.0f}min ({per_model_budget:.0f}s/model)"
    else:
        per_model_budget = 600.0
        budget_note = ""

    n = len(todo)
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {n} models × {epochs} epochs" + budget_note)
    print(f"  workers={workers}, device={device or 'config priority'}")
    print(f"{'='*70}\n")

    def _run_one(model):
        """Train one model; returns (model, elapsed, ok, note)."""
        train_py = os.path.join(model.folder_path, 'train.py')
        if not os.path.exists(train_py):
            return model, 0.0, False, "SKIP (no train.py)"

        cmd = [sys.executable, 'train.py', '--epochs', str(epochs), '--seed', str(seed)]
        if lr is not None:
            cmd += ['--lr', str(lr)]
        if batch_size is not None:
            cmd += ['--batch-size', str(batch_size)]
        if device:
            cmd += ['--device', device]

        t0 = time.time()
        eff_epochs = epochs
        for attempt in range(retries + 2):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=per_model_budget, cwd=model.folder_path)
                elapsed = time.time() - t0
                if result.returncode == 0:
                    updated = _parse_model(model.folder_path, model.index, model.folder_name)
                    return updated, elapsed, True, ""
                # Failed: retry unless we exhausted attempts
                if attempt >= retries:
                    return (model, elapsed, False,
                            f"FAIL ({result.stderr[-200:] if result.stderr else 'exit ' + str(result.returncode)})")
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                if eff_epochs >= 2:
                    eff_epochs //= 2
                    cmd[cmd.index('--epochs') + 1] = str(eff_epochs)
                    continue  # retry with half the epochs within the same budget
                return model, elapsed, False, f"TIMEOUT (>{per_model_budget:.0f}s)"
        return model, time.time() - t0, False, "FAIL (retries exhausted)"

    results = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, n))) as pool:
            futures = {pool.submit(_run_one, m): m for m in todo}
            for fut in as_completed(futures):
                model, elapsed, ok, note = fut.result()
                status = f"OK ({elapsed:.1f}s)  {model.best_metric_name}={model.best_metric_value:.4f}" \
                    if ok else f"{note} ({elapsed:.1f}s)"
                print(f"  [{len(results) + 1}/{n}] {model.folder_name:<35s} {status}")
                if not ok:
                    model.status = 'error'
                    model.error_msg = note
                results.append((model, elapsed, ok))
    else:
        for i, model in enumerate(todo, 1):
            print(f"  [{i}/{n}] {model.folder_name:<35s} ", end='', flush=True)
            model, elapsed, ok, note = _run_one(model)
            if ok:
                print(f"OK ({elapsed:.1f}s)  {model.best_metric_name}={model.best_metric_value:.4f}")
            else:
                print(f"{note} ({elapsed:.1f}s)")
                model.status = 'error'
                model.error_msg = note
            results.append((model, elapsed, ok))

    # ── Leaderboard ──
    ok_results = [r for r in results if r[2]]
    if not ok_results:
        return "\nNo models completed training."

    _LOWER_BETTER = ('loss', 'recon_loss', 'g_loss', 'd_loss', 'val_loss')

    def _sort_key(item):
        m = item[0]
        nm = m.best_metric_name or 'loss'
        val = m.best_metric_value or 0
        if nm in _LOWER_BETTER:
            return val
        return -val
    ok_results.sort(key=_sort_key)

    lines = [
        "",
        f"{'='*70}",
        f"  BENCHMARK RESULTS — {len(ok_results)}/{n} models × {epochs} epochs",
        f"{'='*70}",
        "",
        f"{'Rank':>4s}  {'Model':<30s}  {'Params':>10s}  {'Metric':<20s}  {'Time':>8s}",
        f"{'-'*4}  {'-'*30}  {'-'*10}  {'-'*20}  {'-'*8}",
    ]
    for rank, (m, elapsed, _) in enumerate(ok_results, 1):
        metric_str = f"{m.best_metric_name}={m.best_metric_value:.4f}"
        lines.append(
            f"{rank:4d}  {m.folder_name:<30s}  {m.params:>10,}  {metric_str:<20s}  {elapsed:>7.1f}s"
        )

    # Also write report
    report_path = os.path.join(directory, 'benchmark_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Report\n\n")
        f.write(f"**Epochs**: {epochs}  \n")
        f.write(f"**Models**: {len(ok_results)} trained\n\n")
        f.write(f"| Rank | Model | Params | Metric | Time |\n")
        f.write(f"|------|-------|--------|--------|------|\n")
        for rank, (m, elapsed, _) in enumerate(ok_results, 1):
            metric_str = f"{m.best_metric_name}={m.best_metric_value:.4f}"
            f.write(f"| {rank} | {m.folder_name} | {m.params:,} | {metric_str} | {elapsed:.1f}s |\n")
        f.write(f"\n*Generated by `netgen benchmark`*\n")

    # ── Loss curve chart (best effort; requires matplotlib) ──
    curve_results = [{'model': m, 'elapsed': el} for m, el, ok in ok_results]
    curve_path = _plot_benchmark_curves(curve_results, directory)
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
