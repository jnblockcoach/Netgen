"""Tests for the model manager: scanning, parsing, compare, clean, export."""
import os
import subprocess
import sys

from netgen import find_candidates, gen_folder
from netgen.manager import (benchmark_models, clean_models, compare_models,
                            eval_model, export_models, info_model,
                            list_models, scan_models, sweep_model,
                            train_model, _parse_log)


def _make_model(scratch, index=1, params_range=(5000, 20000), seed=7):
    desc, code, params, inp, outp, mtype = find_candidates(
        *params_range, 1, seed=seed, arch_filter=["mlp"])[0]
    return gen_folder(scratch, index, desc, code, "M{}", params, inp, outp, mtype)


def test_scan_empty_dir(scratch):
    assert scan_models(scratch) == []


def test_scan_finds_generated(scratch):
    folder = _make_model(scratch)
    models = scan_models(scratch)
    assert len(models) == 1
    m = models[0]
    assert m.status == 'generated'
    assert m.params > 0
    assert m.index == 1
    assert m.folder_name == os.path.basename(folder)


def test_scan_status_after_train(scratch):
    folder = _make_model(scratch)
    subprocess.run([sys.executable, "train.py", "--epochs", "1"],
                   cwd=folder, capture_output=True, timeout=300, check=True)
    m = scan_models(scratch)[0]
    assert m.status == 'trained'
    assert m.has_best
    assert m.best_metric_name in ('val_acc', 'val_loss', 'accuracy', 'loss')


def test_parse_log():
    text = ("| Epoch | Loss | Accuracy |\n"
            "|-------|------|----------|\n"
            "| 0 | 1.1000 | 0.5000 |\n"
            "| 1 | 0.9000 | 0.6000 |\n")
    history = _parse_log(text)
    assert len(history) == 2
    assert history[1]["loss"] == 0.9
    assert history[1]["accuracy"] == 0.6


def test_list_models_output(scratch):
    _make_model(scratch)
    out = list_models(scratch)
    assert "1 total" in out
    assert "generated" in out


def test_info_model(scratch):
    _make_model(scratch)
    out = info_model(scratch, "1")
    assert "001-mlp" in out
    assert "Parameters:" in out


def test_compare_sorted_by_params(scratch):
    for i in (1, 2):
        _make_model(scratch, index=i, seed=7 + i)
    out = compare_models(scratch, "params")
    lines = [l for l in out.split("\n") if l.strip().startswith(("1", "2"))]
    # first rank should be the model with fewer params
    assert lines[0].strip().startswith("1")


def test_clean_dry_run_keeps_files(scratch):
    _make_model(scratch)
    out = clean_models(scratch, untrained=True, dry_run=True)
    assert "DRY RUN" in out
    assert len(os.listdir(scratch)) == 1  # folder untouched


def test_clean_force_removes(scratch):
    folder = _make_model(scratch)
    out = clean_models(scratch, untrained=True, dry_run=False)
    assert "Removing 1 models" in out
    assert not os.path.exists(folder)


def test_export_md(scratch):
    _make_model(scratch)
    out = export_models(scratch, "md", os.path.join(scratch, "report.md"))
    assert "Exported" in out
    assert os.path.exists(os.path.join(scratch, "report.md"))


def test_export_json(scratch):
    _make_model(scratch)
    import json
    export_models(scratch, "json", os.path.join(scratch, "report.json"))
    data = json.load(open(os.path.join(scratch, "report.json")))
    assert data[0]["params"] > 0


def test_train_model_command(scratch):
    folder = _make_model(scratch)
    out = train_model(scratch, "1", epochs=1, device="cpu")
    assert "training OK" in out
    assert scan_models(scratch)[0].status == 'trained'


def test_train_unknown_model(scratch):
    out = train_model(scratch, "999")
    assert "not found" in out


def test_train_device_override_works(scratch):
    # model generated with cuda,mps priority; override to cpu at train time
    folder = _make_model(scratch)
    out = train_model(scratch, "1", epochs=1, device="cpu")
    assert "training OK" in out


def test_eval_model_command(scratch):
    folder = _make_model(scratch)
    train_model(scratch, "1", epochs=1, device="cpu")
    out = eval_model(scratch, "1")
    assert "evaluation OK" in out


def test_eval_untrained_model(scratch):
    _make_model(scratch)
    out = eval_model(scratch, "1")
    assert "not trained" in out


def test_benchmark_workers_force(scratch):
    _make_model(scratch)
    _make_model(scratch, index=2)
    out = benchmark_models(scratch, epochs=1, workers=2, force=True, device="cpu")
    assert "BENCHMARK RESULTS" in out
    report = os.path.join(scratch, 'benchmark_report.md')
    assert os.path.exists(report)
    content = open(report, encoding='utf-8').read()
    assert "2 trained" in content or "2/2 models" in content


def test_benchmark_time_budget_no_crash(scratch):
    _make_model(scratch)
    # Tiny budget: models time out but the benchmark must still return
    out = benchmark_models(scratch, epochs=1, time_budget=0.001, device="cpu")
    assert isinstance(out, str) and out.strip() != ""


def test_sweep_updates_config(scratch):
    folder = _make_model(scratch)
    out = sweep_model(scratch, "1", epochs=1, lrs=[0.01], batches=[32],
                      device="cpu")
    assert "SWEEP RESULTS" in out
    cfg = open(os.path.join(folder, 'config.py'), encoding='utf-8').read()
    assert 'LR = 0.01' in cfg and 'BATCH_SIZE = 32' in cfg
    report = os.path.join(scratch, 'sweep_report.md')
    assert os.path.exists(report)


def test_sweep_unknown_model(scratch):
    out = sweep_model(scratch, "999")
    assert "not found" in out


def test_best_metric_falls_back_when_val_all_nan(scratch):
    """Models without val metrics (e.g. GCN) log NaN in val columns."""
    folder = _make_model(scratch)
    with open(os.path.join(folder, 'training_log.md'), 'w', encoding='utf-8') as f:
        f.write("| Epoch | Loss | Accuracy | Val Loss | Val Acc |\n"
                "|-------|------|----------|----------|--------|\n"
                "|     0 | 0.5 | 0.30 | nan | nan |\n"
                "|     1 | 0.4 | 0.35 | nan | nan |\n")
    m = scan_models(scratch)[0]
    assert m.best_metric_name == 'accuracy' and m.best_metric_value == 0.35
    # no NaN anywhere (JSON-export safe)
    for h in m.history:
        for v in h.values():
            assert v == v


def test_benchmark_empty_dir_message(scratch):
    out = benchmark_models(scratch, epochs=1)
    assert "No models found" in out


def test_compare_sort_val_metrics(scratch):
    a = _make_model(scratch, index=1)
    b = _make_model(scratch, index=2)
    for folder, acc in ((a, 0.5), (b, 0.9)):
        with open(os.path.join(folder, 'training_log.md'), 'w', encoding='utf-8') as f:
            f.write("| Epoch | Loss | Accuracy | Val Loss | Val Acc |\n"
                    "|-------|------|----------|----------|--------|\n"
                    f"|     0 | 0.4 | 0.3 | 0.8 | {acc} |\n")
    out = compare_models(scratch, sort_by='val_acc')
    lines = [l for l in out.splitlines() if l.strip().startswith('1')]
    assert 'val_acc=0.9000' in lines[0]  # higher-val-acc model ranks first
