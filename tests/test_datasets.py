"""Dataset integration tests: real datasets, dimension rewriting, compatibility."""
import os

import pytest

from netgen.generator import gen_folder, IMAGE_2D_ARCHS, IMAGE_DATASETS


def _gen(scratch, dataset, arch, index=1, lo=1000, hi=20000, seed=3):
    from netgen import find_candidates
    cands = find_candidates(lo, hi, 1, seed, [arch])
    assert cands, f"no candidate for {arch}"
    desc, code, params, inp, outp, mtype = cands[0]
    return gen_folder(scratch, index, desc, code, "M{}", params, inp, outp,
                      mtype, dataset=dataset, device_priority=['cpu'])


def _first_linear(model_code: str) -> str:
    import re
    m = re.search(r'nn\.(Linear|Conv2d|LSTM|GRU)\(\s*(\d+)', model_code)
    return f"{m.group(1)}({m.group(2)}"


def test_iris_mlp_input_rewritten(scratch):
    folder = _gen(scratch, 'iris', 'mlp')
    model = open(os.path.join(folder, 'model.py'), encoding='utf-8').read()
    assert _first_linear(model).startswith('Linear(4')
    cfg = open(os.path.join(folder, 'config.py'), encoding='utf-8').read()
    assert 'INPUT_DIM = 4' in cfg and 'OUTPUT_DIM = 3' in cfg


def test_iris_excludes_cnn(scratch):
    from netgen.cli import _resolve_arch_filter

    class Opts:
        preset = None
        arch = None

    compat = _resolve_arch_filter(Opts(), 'iris')
    for bad in IMAGE_2D_ARCHS | {'gpt', 't5', 'gcn', 'gan', 'lstm', 'gru',
                                 'bilstm', 'rnn', 'attnlstm', 'ae', 'sae',
                                 'vae', 'contrastive', 'siamese', 'multitask',
                                 'selfattn', 'transformer', 'highway', 'vit', 'mixer'}:
        assert bad not in compat, f"{bad} should be excluded from iris"


def test_mnist_cnn_keeps_image_shape(scratch):
    folder = _gen(scratch, 'mnist', 'cnn')
    model = open(os.path.join(folder, 'model.py'), encoding='utf-8').read()
    assert _first_linear(model).startswith('Conv2d(1')  # MNIST: 1 channel
    eval_code = open(os.path.join(folder, 'eval.py'), encoding='utf-8').read()
    assert 'SynData(train=False)' in eval_code  # held-out test split


def test_mnist_mlp_gets_flattened_vector(scratch):
    folder = _gen(scratch, 'mnist', 'mlp')
    model = open(os.path.join(folder, 'model.py'), encoding='utf-8').read()
    assert _first_linear(model).startswith('Linear(784')
    data = open(os.path.join(folder, 'data.py'), encoding='utf-8').read()
    assert '.view(-1)' in data  # samples flattened to 784-D vectors


def test_cifar10_conv2d_three_channels(scratch):
    folder = _gen(scratch, 'cifar10', 'rescnn')
    model = open(os.path.join(folder, 'model.py'), encoding='utf-8').read()
    assert _first_linear(model).startswith('Conv2d(3')
    eval_code = open(os.path.join(folder, 'eval.py'), encoding='utf-8').read()
    assert 'SynData(train=False)' in eval_code


def test_cifar10_mixer_patch_net_ok(scratch):
    # Patch nets keep their RGB patch embedding untouched
    folder = _gen(scratch, 'cifar10', 'mixer', lo=100000, hi=400000)
    model = open(os.path.join(folder, 'model.py'), encoding='utf-8').read()
    assert 'patch_emb = nn.Linear' in model
    eval_code = open(os.path.join(folder, 'eval.py'), encoding='utf-8').read()
    assert 'SynData(train=False)' in eval_code


def test_mnist_excludes_patch_nets():
    from netgen.cli import _resolve_arch_filter

    class Opts:
        preset = None
        arch = None

    compat = _resolve_arch_filter(Opts(), 'mnist')
    assert 'vit' not in compat and 'mixer' not in compat  # hard-coded RGB


def test_incompatible_arch_raises(scratch):
    from netgen import find_candidates
    from netgen.generator import gen_folder
    cands = find_candidates(1000, 20000, 1, 3, ['cnn'])
    desc, code, params, inp, outp, mtype = cands[0]
    with pytest.raises(ValueError, match='incompatible'):
        gen_folder(scratch, 1, desc, code, "M{}", params, inp, outp, mtype,
                   dataset='iris', device_priority=['cpu'])


def test_moe_full_rewrite(scratch):
    # MoE shares in_dim across router/experts/fc — all linears must change
    folder = _gen(scratch, 'iris', 'moe')
    model = open(os.path.join(folder, 'model.py'), encoding='utf-8').read()
    linears = [ln for ln in model.splitlines() if 'nn.Linear(' in ln]
    inputs = [ln.split('nn.Linear(')[1].split(',')[0] for ln in linears]
    assert all(d == '4' for d in inputs), f"moe linears not all rewritten: {inputs}"
