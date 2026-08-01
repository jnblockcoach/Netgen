"""End-to-end device tests: config resolution, train.py --device override,
and a real 1-epoch training smoke test on the current machine.
"""
import os
import subprocess
import sys

from netgen import find_candidates, gen_folder


def _gen(scratch, device=None, arch="mlp", lo=5000, hi=20000):
    desc, code, params, inp, outp, mtype = find_candidates(lo, hi, 1, seed=7,
                                                           arch_filter=[arch])[0]
    return gen_folder(scratch, 1, desc, code, "M{}", params, inp, outp, mtype,
                      device_priority=device)


def _exec_config(folder):
    """Import the generated config.py in an isolated namespace."""
    cfg_path = os.path.join(folder, "config.py")
    ns = {}
    with open(cfg_path) as f:
        src = f.read()
    exec(compile(src, cfg_path, "exec"), ns)
    return ns


def test_config_resolves_cpu(scratch):
    folder = _gen(scratch, device=["cpu"])
    ns = _exec_config(folder)
    assert ns["DEVICE_PRIORITY"] == ["cpu"]
    assert str(ns["DEVICE"]) == "cpu"


def test_config_resolve_device_function(scratch):
    folder = _gen(scratch, device=["cuda", "cpu"])
    ns = _exec_config(folder)
    # resolve_device must be importable (public name) and overridable
    d = ns["resolve_device"](["cpu"])
    assert str(d) == "cpu"


def test_train_has_device_flag(scratch):
    folder = _gen(scratch, device=["cpu"])
    train = open(os.path.join(folder, "train.py")).read()
    assert "--device" in train
    assert "resolve_device()" in train


def test_train_device_override_smoke(scratch):
    """Full pipeline: generate → train 1 epoch with --device cpu."""
    folder = _gen(scratch, device=["cuda", "mps"])  # will fall back to cpu
    result = subprocess.run(
        [sys.executable, "train.py", "--epochs", "1", "--device", "cpu"],
        cwd=folder, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-500:]
    assert os.path.exists(os.path.join(folder, "model.pth"))
    assert os.path.exists(os.path.join(folder, "training_log.md"))
    assert os.path.exists(os.path.join(folder, "best_model.pth"))


def test_eval_smoke(scratch):
    folder = _gen(scratch, device=["cpu"])
    subprocess.run([sys.executable, "train.py", "--epochs", "1"],
                   cwd=folder, capture_output=True, timeout=300, check=True)
    result = subprocess.run([sys.executable, "eval.py"], cwd=folder,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-500:]
