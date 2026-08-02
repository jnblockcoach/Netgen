"""netgen monitor — live resource monitor for python processes.

Watches every python process (train.py, eval.py, sweep.py, ...) and
reports CPU / GPU / memory usage in a live table. When a configured
limit is exceeded it prints a warning — it NEVER kills processes.

Limits are percentages of the whole machine:
    --cpu 70      warn when python processes use >70% of all CPU cores
    --gpu 80      warn when python processes use >80% of GPU memory
    --memory 60   warn when python processes use >60% of system RAM

GPU stats come from pynvml (nvidia-ml-py) if installed, else from the
`nvidia-smi` CLI; without either, the GPU column shows 'n/a'.
"""
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass
class Limits:
    cpu: float = 70.0
    gpu: float = 80.0
    memory: float = 60.0

    def __post_init__(self):
        for name, val in (('cpu', self.cpu), ('gpu', self.gpu), ('memory', self.memory)):
            if not (0 < val <= 100):
                raise ValueError(f"--{name} limit must be in (0, 100], got {val}")


# ── GPU backend ──────────────────────────────────────────────────────

def _gpu_probe_pynvml():
    """pynvml backend: returns (pid->mem_mb, total_mb, card_util%)."""
    import pynvml
    pynvml.nvmlInit()
    per_pid: Dict[int, int] = {}
    total_mb, util = 0, 0
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        total_mb += pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 ** 2)
        util = max(util, pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            pid = p.pid
            per_pid[pid] = per_pid.get(pid, 0) + p.usedGpuMemory // (1024 ** 2)
    return per_pid, total_mb, util


def _gpu_probe_nvidia_smi():
    """nvidia-smi CLI backend: returns (pid->mem_mb, total_mb, card_util%)."""
    import subprocess
    try:
        procs = subprocess.run(
            ['nvidia-smi', '--query-computation-apps=pid,used_gpu_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        cards = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if procs.returncode != 0 or cards.returncode != 0:
        return None
    per_pid: Dict[int, int] = {}
    for line in procs.stdout.splitlines():
        parts = line.split(',')
        if len(parts) == 2:
            try:
                per_pid[int(parts[0].strip())] = int(parts[1].strip())
            except ValueError:
                pass
    total_mb, util = 0, 0
    for line in cards.stdout.splitlines():
        parts = line.split(',')
        if len(parts) == 2:
            try:
                total_mb += int(parts[0].strip())
                util = max(util, int(parts[1].strip()))
            except ValueError:
                pass
    return per_pid, total_mb, util


def get_gpu_probe():
    """Return a probe callable or None if no GPU backend is available."""
    try:
        _gpu_probe_pynvml()
        return _gpu_probe_pynvml
    except ImportError:
        pass
    except Exception:
        pass
    try:
        probe = _gpu_probe_nvidia_smi()
        if probe is not None:
            return _gpu_probe_nvidia_smi
    except Exception:
        pass
    return None


# ── Process discovery ────────────────────────────────────────────────

def _is_python(proc) -> bool:
    try:
        name = (proc.info.get('name') or '').lower()
        cmd = proc.info.get('cmdline') or []
    except Exception:
        return False
    if 'python' in name:
        return True
    if cmd and 'python' in (cmd[0] or '').lower():
        return True
    return False


def _describe(proc) -> str:
    """Short human-readable description of a process (its command line)."""
    try:
        cmd = proc.info.get('cmdline') or []
    except Exception:
        return '?'
    parts = []
    for c in cmd[1:]:
        if c.endswith('.py'):
            parts.append(os.path.basename(c))
        elif not c.startswith('-'):
            parts.append(c)
        else:
            parts.append(c)
    desc = ' '.join(parts) if parts else (cmd[0] if cmd else 'python')
    return desc[:60]


def _machine_cpu_pct(per_core_pcts: List[float]) -> float:
    """Sum of per-core percentages → fraction of the whole machine.

    Each value is already a percentage of one core (100% = one core), so
    the machine-wide share is sum / num_cores.
    """
    cores = os.cpu_count() or 1
    return sum(per_core_pcts) / cores


def warnings_for(cpu_machine: float, gpu_pct: Optional[float],
                 mem_pct: float, limits: Limits) -> List[str]:
    """Pure check: which limits are exceeded? Returns warning strings."""
    warns = []
    if cpu_machine > limits.cpu:
        warns.append(f"CPU {cpu_machine:.1f}% exceeds limit {limits.cpu:g}% "
                     f"(all python processes, machine-wide)")
    if gpu_pct is not None and gpu_pct > limits.gpu:
        warns.append(f"GPU memory {gpu_pct:.1f}% exceeds limit {limits.gpu:g}% "
                     f"(python processes' share of VRAM)")
    if mem_pct > limits.memory:
        warns.append(f"Memory {mem_pct:.1f}% exceeds limit {limits.memory:g}% "
                     f"(all python processes, machine-wide)")
    return warns


def _fmt_row(pid, desc, cpu, mem, gpu_mb) -> str:
    gpu = f"{gpu_mb:>8d}" if gpu_mb is not None else f"{'n/a':>8s}"
    return f"{pid:>7d}  {desc:<44s}  {cpu:>8.1f}%  {mem:>7.1f}%  {gpu}"


def _header(limits: Limits, gpu_ok: bool) -> str:
    gpu_note = "gpu n/a" if not gpu_ok else f"gpu {limits.gpu:g}%"
    return (f" PID       PROCESS                        CPU/core   MEM(machine)   GPU(mb)\n"
            f"{'='*78}\n"
            f" limits: cpu {limits.cpu:g}% (machine)  {gpu_note}  mem {limits.memory:g}%  "
            f"— python processes only, watching (never killing)")


# ── Main loop ────────────────────────────────────────────────────────

def run_monitor(cpu_limit: float = 70.0, gpu_limit: float = 80.0,
                memory_limit: float = 60.0, interval: float = 2.0,
                duration: float = 0.0, once: bool = False,
                pids: Optional[List[int]] = None) -> str:
    """Run the monitor until Ctrl+C / duration / one sample.

    Returns the final screen text (useful for tests and --once).
    """
    if psutil is None:
        raise SystemExit("`netgen monitor` requires psutil — install it with: "
                         "pip install psutil")

    limits = Limits(cpu_limit, gpu_limit, memory_limit)
    interval = max(0.2, float(interval))
    pids = set(pids or [])
    gpu_probe = get_gpu_probe()

    procs: Dict[int, object] = {}   # pid -> psutil.Process (keeps cpu sample base)
    warn_count = 0
    start = time.time()
    tty = sys.stdout.isatty()
    screens = []
    primed = not once  # --once: first pass only primes the cpu sample base

    try:
        while True:
            # 1. refresh process table
            try:
                snapshot = list(psutil.process_iter(['pid', 'name', 'cmdline',
                                                     'memory_info']))
            except Exception:
                snapshot = []
            current: Dict[int, object] = {}
            rows = []
            total_cpu = 0.0
            total_mem = 0.0
            for proc in snapshot:
                pid = proc.info['pid']
                if pid == os.getpid():
                    continue
                if pids and pid not in pids:
                    continue
                if not _is_python(proc):
                    continue
                try:
                    handle = procs.get(pid) or psutil.Process(pid)
                    cpu = handle.cpu_percent(interval=None)  # per core, 100% = 1 core
                    mem = handle.memory_info().rss / psutil.virtual_memory().total * 100
                    current[pid] = handle
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                rows.append((pid, _describe(proc), cpu, mem))
                total_cpu += cpu
                total_mem += mem
            procs = current

            # 2. GPU stats
            gpu_per_pid: Dict[int, int] = {}
            gpu_total_mb = 0
            gpu_util = None
            if gpu_probe is not None:
                try:
                    gpu_per_pid, gpu_total_mb, gpu_util = gpu_probe()
                except Exception:
                    gpu_probe = None

            # 3. build the screen
            cpu_machine = _machine_cpu_pct([r[2] for r in rows])
            gpu_mb_used = sum(gpu_per_pid.get(pid, 0) for pid, *_ in rows)
            gpu_pct = (gpu_mb_used / gpu_total_mb * 100.0
                       if gpu_probe is not None and gpu_total_mb > 0 else None)
            warns = warnings_for(cpu_machine, gpu_pct, total_mem, limits)
            if warns:
                warn_count += 1

            if not primed:
                # --once: first pass is the CPU baseline; sample again after
                # one interval so percentages are meaningful.
                primed = True
                time.sleep(interval)
                continue

            lines = [_header(limits, gpu_probe is not None)]
            for pid, desc, cpu, mem in sorted(rows, key=lambda r: -r[2]):
                lines.append(_fmt_row(pid, desc, cpu, mem,
                                      gpu_per_pid.get(pid)))
            lines.append('-' * 78)
            total_gpu = (f"{gpu_pct:.1f}% ({gpu_mb_used} MB)" if gpu_pct is not None
                         else "n/a")
            lines.append(
                f"TOTAL python:  CPU {cpu_machine:.1f}% of machine   "
                f"MEM {total_mem:.1f}%   GPU {total_gpu}")
            if gpu_util is not None and gpu_probe is not None:
                lines.append(f"GPU cards utilization: {gpu_util}% (all processes, incl. non-python)")
            if warns:
                lines.append("")
                lines.append("\033[91m⚠ WARNING\033[0m" if tty else "⚠ WARNING")
                for w in warns:
                    lines.append("  - " + w)
                lines.append("  Training is NOT interrupted — the monitor only watches. "
                             "Consider --workers 1, fewer concurrent trainings, or a smaller batch.")
            lines.append(f"\nrefreshing every {interval:g}s · warnings so far: {warn_count} · Ctrl+C to quit")

            screen = '\n'.join(lines)
            screens.append(screen)
            if tty:
                sys.stdout.write('\033[H\033[2J' + screen + '\n')
            else:
                sys.stdout.write(screen + '\n' + '-' * 60 + '\n')
            sys.stdout.flush()

            if once:
                break
            if duration and time.time() - start >= duration:
                break
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nmonitor stopped — {warn_count} warning(s) issued, no process was killed")

    return screens[-1] if screens else ""
