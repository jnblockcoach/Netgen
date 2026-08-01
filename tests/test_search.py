"""Tests for candidate search (find_candidates / list_architectures)."""
import pytest

from netgen import find_candidates, list_architectures


def test_count_and_range():
    cands = find_candidates(10_000, 20_000, 10, seed=42)
    assert len(cands) == 10
    for _, _, params, _, _, _ in cands:
        assert 10_000 <= params <= 20_000


def test_descriptions_unique():
    cands = find_candidates(10_000, 20_000, 20, seed=42)
    descs = [c[0] for c in cands]
    assert len(descs) == len(set(descs))


def test_seed_reproducible():
    a = find_candidates(10_000, 20_000, 10, seed=7)
    b = find_candidates(10_000, 20_000, 10, seed=7)
    assert [(c[0], c[2]) for c in a] == [(c[0], c[2]) for c in b]


def test_different_seed_differs():
    a = find_candidates(10_000, 20_000, 10, seed=1)
    b = find_candidates(10_000, 20_000, 10, seed=2)
    assert [(c[0], c[2]) for c in a] != [(c[0], c[2]) for c in b]


def test_arch_filter_respected():
    cands = find_candidates(10_000, 20_000, 8, seed=42, arch_filter=["mlp", "lstm"])
    archs = {c[0].split("-")[0] for c in cands}
    assert archs <= {"mlp", "lstm"}


def test_arch_filter_all_archs():
    all_archs = list_architectures()
    assert len(all_archs) == 31
    cands = find_candidates(10_000, 20_000, 5, seed=1, arch_filter=all_archs)
    assert cands


def test_bad_range():
    with pytest.raises(ValueError):
        find_candidates(0, -5, 1)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        find_candidates(1000, 5000, 1, arch_filter=["nope"])


def test_fixed_input_output():
    cands = find_candidates(5000, 20_000, 5, seed=3,
                            arch_filter=["mlp", "deep"],
                            fixed_input=64, fixed_output=3)
    for _, _, _, inp, outp, _ in cands:
        assert inp == 64
        assert outp == 3


def test_sorted_by_params():
    cands = find_candidates(5000, 30_000, 10, seed=5)
    cands.sort(key=lambda c: c[2])
    params = [c[2] for c in cands]
    assert params == sorted(params)


def test_negative_range_rejected():
    with pytest.raises(ValueError):
        find_candidates(-10, 100, 1)
