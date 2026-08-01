"""CLI-level tests: argument parsing + end-to-end generate/list commands."""
import os

import pytest

from netgen.cli import _parse_device, _parse_range, run


# ── parsing helpers ──

def test_parse_range_ok():
    assert _parse_range("10000-20000") == (10000, 20000)


@pytest.mark.parametrize("bad", ["abc", "1-2-3", "20-10", "-5-10", "5--3"])
def test_parse_range_bad(bad):
    with pytest.raises(ValueError):
        _parse_range(bad)


def test_parse_device_default():
    assert _parse_device(None) == ["cuda", "mps"]
    assert _parse_device("") == ["cuda", "mps"]


def test_parse_device_ok():
    assert _parse_device("cuda,mps") == ["cuda", "mps"]
    assert _parse_device("CPU") == ["cpu"]  # case-insensitive
    assert _parse_device(" cuda , cpu ") == ["cuda", "cpu"]


def test_parse_device_invalid():
    with pytest.raises(ValueError):
        _parse_device("cuda,tpu")


# ── end-to-end CLI ──

def test_generate_and_list(scratch, capsys):
    out_dir = os.path.join(scratch, "models")
    rc = run(["generate", "--range", "5000-20000", "--count", "2",
              "--arch", "mlp", "--device", "cpu", "-o", out_dir])
    assert rc == 0
    folders = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    assert len(folders) == 2

    rc = run(["list", "--dir", out_dir])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 total" in out


def test_generate_invalid_device(scratch):
    rc = run(["generate", "--range", "5000-20000", "--count", "1",
              "--device", "npu", "-o", scratch])
    assert rc == 1


def test_generate_bad_range(scratch):
    rc = run(["generate", "--range", "20000-10000", "--count", "1", "-o", scratch])
    assert rc == 1


def test_generate_old_style_flags(scratch):
    # backward compat: first arg starts with -- → treated as generate
    rc = run(["--range", "5000-20000", "--count", "1", "--arch", "mlp", "-o", scratch])
    assert rc == 0
    assert any("001-" in d for d in os.listdir(scratch))


def test_info_unknown_model(scratch, capsys):
    rc = run(["info", "999", "--dir", scratch])
    assert rc == 0
    out = capsys.readouterr().out
    assert "not found" in out


def test_archs_list(capsys):
    rc = run(["archs", "--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Total: 31 architectures" in out
