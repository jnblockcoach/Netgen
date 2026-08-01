"""Verify that every architecture generator produces code whose *actual*
instantiated parameter count matches the declared count.

This is the core "verified parameter counts" guarantee of NetGen.
"""
import torch

from netgen import count_params, find_candidates, list_architectures
from netgen.search import SAMPLERS

# Range buckets covering all tiers; samplers are skipped in buckets where
# their architecture pool does not apply (medium >=100K, large >=10M).
RANGE_BUCKETS = [
    (1, 200),            # unary (1-param)
    (1000, 90_000),      # quick tier
    (100_000, 9_000_000),  # standard tier
    (10_000_000, 150_000_000),  # production tier (bounded to keep tests fast)
]


def _instantiate(code: str):
    """Compile + exec generated model code, return (module_class, params)."""
    compiled = compile("import torch\nimport torch.nn as nn\n" + code, "<gen>", "exec")
    ns = {}
    exec(compiled, ns)
    return ns["M1"]


def _candidates_for(arch: str, count: int = 2):
    """Find candidates for one architecture across range buckets."""
    for lo, hi in RANGE_BUCKETS:
        cands = find_candidates(lo, hi, count, seed=42, arch_filter=[arch])
        if cands:
            return cands
    return []


def test_all_architectures_instantiable_and_param_exact():
    """Declared params must equal count_params of the instantiated module."""
    archs = sorted(SAMPLERS.keys())
    assert len(archs) == 31

    failures = []
    for arch in archs:
        cands = _candidates_for(arch)
        assert cands, f"{arch}: no candidate found in any range bucket"
        for desc, code, declared, inp, outp, mtype in cands:
            try:
                cls = _instantiate(code.format(1))
                m = cls()
                actual = count_params(m)
            except Exception as e:
                failures.append(f"{arch} ({desc}): instantiation failed: {e}")
                continue
            if actual != declared:
                failures.append(
                    f"{arch} ({desc}): declared {declared:,} != actual {actual:,}"
                )
    assert not failures, "Parameter count mismatches:\n" + "\n".join(failures)


def test_declared_params_positive():
    """Every candidate must declare a positive parameter count."""
    for arch in sorted(SAMPLERS.keys()):
        for _, _, params, _, _, _ in _candidates_for(arch):
            assert params > 0, f"{arch}: non-positive params {params}"


def test_generated_code_compiles_for_every_arch():
    """Code templates must be syntactically valid after class-name fill."""
    for arch in sorted(SAMPLERS.keys()):
        for _, code, _, _, _, _ in _candidates_for(arch):
            compile("import torch\nimport torch.nn as nn\n" + code.format(1),
                    f"<{arch}>", "exec")


def test_unary_has_exactly_one_param():
    cands = find_candidates(1, 10, 3, seed=1, arch_filter=["unary"])
    assert cands
    for desc, code, params, inp, outp, mtype in cands:
        cls = _instantiate(code.format(1))
        assert count_params(cls()) == 1
        assert params == 1
