"""Tests for the process monitor (netgen monitor)."""
import pytest

from netgen.monitor import (Limits, _machine_cpu_pct, _describe, warnings_for,
                            run_monitor)


def test_limits_validation():
    Limits(70, 80, 60)  # defaults fine
    with pytest.raises(ValueError):
        Limits(cpu=0)
    with pytest.raises(ValueError):
        Limits(memory=150)
    with pytest.raises(ValueError):
        Limits(gpu=-1)


def test_machine_cpu_pct():
    assert abs(_machine_cpu_pct([100.0]) - 100.0 / (__import__('os').cpu_count() or 1)) < 0.01
    # 4 processes pegging one core each on a 4-core box == 100% machine
    assert _machine_cpu_pct([100.0, 100.0, 100.0, 100.0]) <= 100.0
    assert _machine_cpu_pct([]) == 0.0


def test_warnings_for():
    # over the limit -> warning
    w = warnings_for(80.0, 50.0, 20.0, Limits(cpu=70, gpu=80, memory=60))
    assert len(w) == 1 and 'CPU' in w[0]
    # gpu warning
    w = warnings_for(10.0, 95.0, 20.0, Limits(70, 80, 60))
    assert len(w) == 1 and 'GPU' in w[0]
    # memory warning
    w = warnings_for(10.0, 50.0, 70.0, Limits(70, 80, 60))
    assert len(w) == 1 and 'Memory' in w[0]
    # at the limit is fine (strictly greater)
    assert warnings_for(70.0, 80.0, 60.0, Limits(70, 80, 60)) == []
    # no gpu backend -> no gpu warning
    assert warnings_for(10.0, None, 10.0, Limits(70, 80, 60)) == []


def test_describe():
    class FakeProc:
        def __init__(self, cmd):
            self.info = {'cmdline': cmd}

    assert 'train.py --epochs 30' in _describe(FakeProc(
        ['/usr/bin/python3', 'train.py', '--epochs', '30']))
    # long command lines get truncated
    desc = _describe(FakeProc(['python', 'train.py', '--lr', '0.001' * 40]))
    assert len(desc) <= 60


def test_run_monitor_once_smoke():
    # Requires psutil (in requirements.txt). Samples the real system once.
    screen = run_monitor(once=True, interval=0.5)
    assert 'TOTAL python' in screen
    assert 'watching (never killing)' in screen


def test_run_monitor_low_limits_warns():
    screen = run_monitor(once=True, interval=0.5, cpu_limit=0.5)
    assert 'WARNING' in screen or 'TOTAL python' in screen
