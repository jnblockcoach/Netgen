"""Tests for folder generation: file sets per tier, config content,
generated code validity, and --device integration.
"""
import os
import py_compile

from netgen import find_candidates, gen_folder
from netgen.generator import gen_config, _get_tier

QUICK_FILES = {"config.py", "data.py", "data_explore.py", "model.py", "train.py",
               "eval.py", "predict.py", "visualize.py", "requirements.txt", "README.md"}
STANDARD_FILES = QUICK_FILES | {"sweep.py"}
PRODUCTION_FILES = STANDARD_FILES | {"model/", "configs/", "scripts/"}


def _gen_one(scratch, params_range=(10_000, 20_000), arch="mlp", device=None,
             dataset="syn", seed=7):
    cands = find_candidates(*params_range, 1, seed=seed, arch_filter=[arch])
    assert cands, f"no candidate for {arch} in {params_range}"
    desc, code, params, inp, outp, mtype = cands[0]
    folder = gen_folder(scratch, 1, desc, code, "M{}", params, inp, outp, mtype,
                        dataset=dataset, device_priority=device)
    return folder


def test_quick_file_set(scratch):
    folder = _gen_one(scratch)
    names = set(os.listdir(folder)) | {
        d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))}
    files = {n for n in names if not os.path.isdir(os.path.join(folder, n))}
    dirs = {n for n in names if os.path.isdir(os.path.join(folder, n))}
    assert QUICK_FILES - files == set() or QUICK_FILES - dirs == set()
    # quick tier: no sweep.py
    assert "sweep.py" not in files


def test_standard_file_set(scratch):
    folder = _gen_one(scratch, params_range=(60_000, 90_000))
    assert _get_tier(_params_of(folder)) == 'standard'
    files = {n for n in os.listdir(folder)}
    assert "sweep.py" in files
    assert "predict.py" in files


def test_production_file_set(scratch):
    folder = _gen_one(scratch, params_range=(60_000_000, 90_000_000), arch="gpt")
    assert _get_tier(_params_of(folder)) == 'production'
    assert os.path.isdir(os.path.join(folder, "model"))
    assert os.path.isdir(os.path.join(folder, "configs"))
    assert os.path.isdir(os.path.join(folder, "scripts"))
    for f in ("scripts/benchmark.py", "scripts/profile.py", "scripts/export.py"):
        assert os.path.exists(os.path.join(folder, f))


def test_all_generated_py_compile(scratch):
    folder = _gen_one(scratch)
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".py"):
                py_compile.compile(os.path.join(root, f), doraise=True)


def test_config_device_default(scratch):
    folder = _gen_one(scratch)
    cfg = open(os.path.join(folder, "config.py")).read()
    assert "DEVICE_PRIORITY = ['cuda', 'mps', 'cpu']" in cfg
    assert "DEVICE = " in cfg


def test_config_device_cpu(scratch):
    folder = _gen_one(scratch, device=["cpu"])
    cfg = open(os.path.join(folder, "config.py")).read()
    assert "DEVICE_PRIORITY = ['cpu']" in cfg


def test_config_device_omitted_cpu(scratch):
    # ['cuda', 'cpu'] must normalize to ['cuda', 'cpu'] (cpu appended once)
    folder = _gen_one(scratch, device=["cuda", "cpu"])
    cfg = open(os.path.join(folder, "config.py")).read()
    assert "DEVICE_PRIORITY = ['cuda', 'cpu']" in cfg


def test_train_has_device_migration(scratch):
    folder = _gen_one(scratch)
    train = open(os.path.join(folder, "train.py")).read()
    assert "def _dev(batch)" in train
    assert "map(_dev, lo)" in train
    assert ".to(DEVICE)" in train
    assert "m=M1().to(DEVICE)" in train


def test_eval_has_device_migration(scratch):
    folder = _gen_one(scratch)
    eval_code = open(os.path.join(folder, "eval.py")).read()
    assert "m=M1().to(DEVICE)" in eval_code


def test_model_params_match_readme(scratch):
    folder = _gen_one(scratch)
    model_code = open(os.path.join(folder, "model.py")).read()
    ns = {}
    exec(model_code, ns)
    m = ns["M1"]()
    actual = sum(p.numel() for p in m.parameters())
    readme = open(os.path.join(folder, "README.md")).read()
    import re
    declared = int(re.search(r"\*\*Parameters\*\*:\s*([\d,]+)", readme).group(1).replace(",", ""))
    assert actual == declared


def test_real_dataset_rewrites_dims(scratch):
    folder = _gen_one(scratch, dataset="iris")
    cfg = open(os.path.join(folder, "config.py")).read()
    assert "INPUT_DIM = 4" in cfg
    assert "OUTPUT_DIM = 3" in cfg
    model = open(os.path.join(folder, "model.py")).read()
    ns = {}
    exec(model, ns)
    m = ns["M1"]()
    # input must be 4-dim now
    assert list(m.state_dict().values())[0].shape[1] == 4


def test_gen_config_device_block():
    cfg = gen_config(10, 3, device_priority=["cuda", "mps"])
    assert "DEVICE_PRIORITY = ['cuda', 'mps', 'cpu']" in cfg
    assert "resolve_device" in cfg


def _params_of(folder):
    import re
    readme = open(os.path.join(folder, "README.md")).read()
    return int(re.search(r"\*\*Parameters\*\*:\s*([\d,]+)", readme).group(1).replace(",", ""))


def test_standard_tier_scheduler_guard(scratch):
    """SCHEDULER='none' (default) must not call scheduler.step()."""
    from netgen import find_candidates
    from netgen.generator import gen_folder
    c = find_candidates(80_000, 120_000, 1, seed=4, arch_filter=['mlp'])[0]
    desc, code, params, inp, outp, mtype = c
    folder = gen_folder(scratch, 1, desc, code, "M{}", params, inp, outp,
                        mtype, device_priority=['cpu'])
    train = open(os.path.join(folder, 'train.py'), encoding='utf-8').read()
    assert 'scheduler=None' in train
    assert 'if scheduler is not None: scheduler.step(loss_val)' in train
    assert 'scheduler.step(loss_val)' not in train.replace(
        'if scheduler is not None: scheduler.step(loss_val)', '')


def test_val_split_denominator_uses_train_subset(scratch):
    """Epoch-end validation divides by the training subset, not len(ds)."""
    from netgen import find_candidates
    from netgen.generator import gen_folder
    c = find_candidates(5000, 20000, 1, seed=4, arch_filter=['mlp'])[0]
    desc, code, params, inp, outp, mtype = c
    folder = gen_folder(scratch, 1, desc, code, "M{}", params, inp, outp,
                        mtype, device_priority=['cpu'])
    train = open(os.path.join(folder, 'train.py'), encoding='utf-8').read()
    assert '/len(_train_ds)' in train
    assert ')/len(ds)' not in train.replace(')/len(_train_ds)', '')
