"""Architecture search and candidate generation.

Uses a registry pattern: each architecture registers a sampler function
that knows how to generate random candidates in a parameter range.
"""
import random
from typing import List, Tuple, Optional, Callable

from .architectures import *
from .generator import _rewrite_model_dims


# Candidate type: (description, code, params, input_dim, output_dim, model_type)
Candidate = Tuple[str, str, int, int, int, str]

# Sampler: function(lo, hi) -> Optional[Candidate]
Sampler = Callable[[int, int], Optional[Candidate]]

# ── Helper utilities ──

def _rand_in_range(lo_s: int, hi_s: int, abs_max: int = 100000) -> int:
    """Random integer in [max(1, lo_s), min(abs_max, hi_s)], safe for edge cases."""
    lo = max(1, lo_s)
    hi = min(abs_max, hi_s)
    if lo >= hi:
        return lo
    return random.randint(lo, hi)


def _ok(lo: int, hi: int, params: int) -> bool:
    """Check if params fall within [lo, hi]."""
    return lo <= params <= hi


# ── Architecture samplers ──
# Each sampler generates random dimensions, checks approximate params,
# then builds the actual model. Returns None if out of range.

def _sample_unary(lo: int, hi: int) -> Optional[Candidate]:
    variant = random.choice(["a", "b", "c"])
    if variant == "a":
        code, params, inp, outp, mt = make_unary_a()
        desc = "unary-a-linear"
    elif variant == "b":
        code, params, inp, outp, mt = make_unary_b()
        desc = "unary-b-scalar"
    else:
        code, params, inp, outp, mt = make_unary_c()
        desc = "unary-c-bias"
    if not _ok(lo, hi, params):
        return None
    return desc, code, params, inp, outp, mt


def _sample_linear(lo: int, hi: int) -> Optional[Candidate]:
    max_dim = 20000
    in_feat = _rand_in_range(1, min(max_dim, int(hi ** 0.5)), 20000)
    out_feat = _rand_in_range(1, min(max_dim, int(hi ** 0.5)), 20000)
    params = in_feat * out_feat + out_feat
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_linear(in_feat, out_feat)
    desc = f"linear-{in_feat}x{out_feat}"
    return desc, code, params, inp, outp, mt


def _sample_mlp(lo: int, hi: int, tiny: bool = False) -> Optional[Candidate]:
    if tiny:
        d1, d2, d3 = random.randint(1, 5), random.randint(1, 5), random.randint(1, 3)
    else:
        d1 = _rand_in_range(1, min(20000, int(hi ** 0.55)), 20000)
        d2 = _rand_in_range(1, min(20000, int(hi ** 0.55)), 20000)
        d3 = _rand_in_range(1, min(20000, int(hi ** 0.3)), 5000)
    params = d1 * d2 + d2 + d2 * d3 + d3
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_mlp([d1, d2, d3])
    desc = f"mlp-{d1}x{d2}x{d3}"
    return desc, code, params, inp, outp, mt


def _sample_lstm(lo: int, hi: int) -> Optional[Candidate]:
    hs = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    nl = min(4, max(1, int(hi ** 0.1)))
    # Approximate LSTM params
    params = sum(4 * hs * ((1 if i == 0 else hs) + hs + 2) for i in range(nl)) + hs * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_lstm(1, hs, nl, 10)
    desc = f"lstm-{hs}h-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_gru(lo: int, hi: int) -> Optional[Candidate]:
    hs = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    nl = min(4, max(1, int(hi ** 0.1)))
    params = sum(3 * hs * ((1 if i == 0 else hs) + hs + 2) for i in range(nl)) + hs * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_gru(1, hs, nl, 10)
    desc = f"gru-{hs}h-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_bilstm(lo: int, hi: int) -> Optional[Candidate]:
    hs = _rand_in_range(1, min(20000, int(hi ** 0.42)), 20000)
    nl = min(3, max(1, int(hi ** 0.1)))
    code, params, inp, outp, mt = make_bilstm(1, hs, nl, 10)
    if not _ok(lo, hi, params):
        return None
    desc = f"bilstm-{hs}h-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_cnn(lo: int, hi: int) -> Optional[Candidate]:
    oc = _rand_in_range(1, min(50000, int(hi ** 0.5)), 50000)
    params = 1 * oc * 9 + oc + oc * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_cnn(1, [(oc, 0)], [], 10)
    desc = f"cnn-{oc}f"
    return desc, code, params, inp, outp, mt


def _sample_ae(lo: int, hi: int, tiny: bool = False) -> Optional[Candidate]:
    if tiny:
        dim = random.randint(2, max(5, hi))
        hidden = random.randint(1, max(1, hi // 2))
    else:
        dim = _rand_in_range(1, min(100000, int(hi ** 0.6)), 100000)
        hidden = _rand_in_range(1, min(50000, int(hi ** 0.5)), 50000)
    params = 2 * (dim * hidden + hidden)
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_ae(dim, hidden)
    desc = f"ae-{dim}x{hidden}"
    return desc, code, params, inp, outp, mt


def _sample_vae(lo: int, hi: int) -> Optional[Candidate]:
    di = _rand_in_range(2, min(50000, int(hi ** 0.4)), 50000)
    dh = _rand_in_range(2, min(50000, int(hi ** 0.4)), 50000)
    dl = _rand_in_range(1, min(20000, int(hi ** 0.3)), 20000)
    code, params, inp, outp, mt = make_vae(di, dh, dl)
    if not _ok(lo, hi, params):
        return None
    desc = f"vae-{di}x{dh}x{dl}"
    return desc, code, params, inp, outp, mt


def _sample_resblock(lo: int, hi: int) -> Optional[Candidate]:
    n = _rand_in_range(1, 20, 50)
    d = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    h = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    params = n * (d * h + h + h * d + d) + d * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_resblock(d, h, n, 10)
    desc = f"resblock-{d}x{h}x{n}"
    return desc, code, params, inp, outp, mt


def _sample_highway(lo: int, hi: int) -> Optional[Candidate]:
    d = _rand_in_range(1, min(50000, int(hi ** 0.45)), 50000)
    n = _rand_in_range(1, min(100, int(hi ** 0.25)), 200)
    params = n * 2 * (d * d + d) + d * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_highway(d, n, 10)
    desc = f"highway-{d}x{n}"
    return desc, code, params, inp, outp, mt


def _sample_moe(lo: int, hi: int) -> Optional[Candidate]:
    d = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    h = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    e = _rand_in_range(1, 64, 128)
    params = d * e + e + e * (2 * d * h + d + h) + d * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_moe(d, h, e, 10)
    desc = f"moe-{d}x{h}x{e}"
    return desc, code, params, inp, outp, mt


def _sample_multitask(lo: int, hi: int) -> Optional[Candidate]:
    d = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    h = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    params = d * h + h + 2 * (h * 10 + 10)
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_multitask(d, h, 2, 10)
    desc = f"multitask-{d}x{h}"
    return desc, code, params, inp, outp, mt


def _sample_gan(lo: int, hi: int) -> Optional[Candidate]:
    zd = _rand_in_range(1, 100, 500)
    gh = _rand_in_range(1, min(20000, int(hi ** 0.4)), 50000)
    dh = _rand_in_range(1, min(10000, int(hi ** 0.35)), 20000)
    dd = 2  # data_dim
    params = zd * gh + gh + gh * dd + dd + dd * dh + dh + dh * 1 + 1
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_gan(zd, gh, dh, dd)
    desc = f"gan-z{zd}-g{gh}-d{dh}"
    return desc, code, params, inp, outp, mt


def _sample_contrastive(lo: int, hi: int) -> Optional[Candidate]:
    d = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    h = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    params = d * h + h + h * 64 + 64
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_contrastive(d, h, 64)
    desc = f"contrast-{d}x{h}"
    return desc, code, params, inp, outp, mt


def _sample_siamese(lo: int, hi: int) -> Optional[Candidate]:
    d = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    h = _rand_in_range(1, min(20000, int(hi ** 0.45)), 20000)
    params = d * h + h + h * 32 + 32
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_siamese(d, h, 32)
    desc = f"siamese-{d}x{h}"
    return desc, code, params, inp, outp, mt


def _sample_deep(lo: int, hi: int, max_dim: int = 100000,
                 max_layers: int = 100) -> Optional[Candidate]:
    # Scale limits for extremely large ranges (trillions+)
    _max_dim = max(max_dim, min(10_000_000, int(hi ** 0.45)))
    _max_layers = max(max_layers, min(100_000, int(hi ** 0.22)))
    d = _rand_in_range(2, min(_max_dim, int(hi ** 0.48)), _max_dim)
    nl = _rand_in_range(2, min(_max_layers, int(hi ** 0.25)), _max_layers)
    params = d * d + d + (nl - 2) * (d * d + d) + d * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_deep_mlp(d, nl, 10)
    desc = f"deep-{d}x{nl}"
    return desc, code, params, inp, outp, mt


def _sample_sae(lo: int, hi: int, max_dim: int = 100000,
                max_layers: int = 25) -> Optional[Candidate]:
    _max_dim = max(max_dim, min(10_000_000, int(hi ** 0.45)))
    _max_layers = max(max_layers, min(50_000, int(hi ** 0.18)))
    d = _rand_in_range(2, min(_max_dim, int(hi ** 0.5)), _max_dim)
    nl = _rand_in_range(1, min(_max_layers, int(hi ** 0.2)), _max_layers)
    code, params, inp, outp, mt = make_stacked_ae(d, nl, 4)
    if not _ok(lo, hi, params):
        return None
    desc = f"sae-{d}x{nl}"
    return desc, code, params, inp, outp, mt


def _sample_wide(lo: int, hi: int, max_dim: int = 100000) -> Optional[Candidate]:
    _max_dim = max(max_dim, min(10_000_000, int(hi ** 0.45)))
    _max_hidden = max(200000, min(20_000_000, int(hi ** 0.5)))
    in_d = _rand_in_range(1, min(_max_dim, int(hi ** 0.45)), _max_dim)
    h_d = _rand_in_range(1, min(_max_hidden, int(hi ** 0.55)), _max_hidden)
    params = in_d * h_d + h_d + h_d * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_wide_net(in_d, h_d, 10)
    desc = f"wide-{in_d}x{h_d}"
    return desc, code, params, inp, outp, mt


def _sample_transformer(lo: int, hi: int, max_dim: int = 100000,
                        max_layers: int = 50) -> Optional[Candidate]:
    _max_dim = max(max_dim, min(10_000_000, int(hi ** 0.42)))
    _max_layers = max(max_layers, min(50_000, int(hi ** 0.2)))
    dm = _rand_in_range(2, min(_max_dim, int(hi ** 0.42)), _max_dim)
    nl = _rand_in_range(1, min(_max_layers, int(hi ** 0.22)), _max_layers)
    nh = max(1, dm // 64)
    code, params, inp, outp, mt = make_transformer(dm, nh, nl)
    if not _ok(lo, hi, params):
        return None
    desc = f"tf-{dm}x{nl}"
    return desc, code, params, inp, outp, mt


# ═══════════════════════════════════════════════════
#  Medium-tier samplers (≥ 100K params)
# ═══════════════════════════════════════════════════

def _sample_rescnn(lo: int, hi: int) -> Optional[Candidate]:
    """ResCNN: multi-stage residual CNN."""
    in_ch = random.choice([1, 3])
    num_stages = _rand_in_range(2, min(5, int(hi ** 0.15)), 10)
    base_ch = _rand_in_range(8, min(512, int(hi ** 0.4)), 512)
    blocks_per_stage = _rand_in_range(1, min(4, int(hi ** 0.1)), 8)
    stages = []
    ch = base_ch
    for _ in range(num_stages):
        stages.append((ch, blocks_per_stage))
        ch = min(ch * 2, 1024)
    # Approx: each block = ch_in*ch*9 + ch + ch*ch*9 + ch + 4*ch
    params = in_ch * base_ch * 9 + base_ch * 2
    prev = base_ch
    for out_ch, nb in stages:
        for b in range(nb):
            c_in = prev if b == 0 else out_ch
            params += c_in * out_ch * 9 + out_ch + out_ch * out_ch * 9 + out_ch + out_ch * 4
            prev = out_ch
    params += prev * 10 + 10  # fc
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_rescnn(in_ch, stages, 10)
    desc = f"rescnn-{in_ch}ch-{num_stages}s-{blocks_per_stage}b"
    return desc, code, params, inp, outp, mt


def _sample_sepcnn(lo: int, hi: int) -> Optional[Candidate]:
    in_ch = random.choice([1, 3])
    num_layers = _rand_in_range(2, min(10, int(hi ** 0.2)), 15)
    channels = []
    ch = _rand_in_range(16, min(256, int(hi ** 0.35)), 256)
    for _ in range(num_layers):
        channels.append(ch)
        ch = min(ch * 2, 1024)
    params = 0
    prev = in_ch
    for c in channels:
        params += prev * 9 + prev + prev * c + c + c * 2
        prev = c
    params += prev * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_sepcnn(in_ch, channels, 10)
    desc = f"sepcnn-{in_ch}ch-{num_layers}l"
    return desc, code, params, inp, outp, mt


def _sample_densecnn(lo: int, hi: int) -> Optional[Candidate]:
    in_ch = random.choice([1, 3])
    growth = _rand_in_range(4, min(64, int(hi ** 0.3)), 64)
    num_layers = _rand_in_range(3, min(20, int(hi ** 0.2)), 30)
    params = in_ch * growth * 2 * 9 + growth * 2
    prev_total = growth * 2
    for i in range(num_layers):
        params += prev_total * 2 + growth + prev_total * growth * 9 + growth
        prev_total += growth
    params += prev_total * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_densecnn(in_ch, growth, num_layers, 10)
    desc = f"densecnn-{in_ch}ch-g{growth}-{num_layers}l"
    return desc, code, params, inp, outp, mt


def _sample_attnlstm(lo: int, hi: int) -> Optional[Candidate]:
    hs = _rand_in_range(16, min(1024, int(hi ** 0.4)), 1024)
    nl = _rand_in_range(1, min(4, int(hi ** 0.12)), 6)
    nh = max(1, hs // 64)
    params = sum(4 * hs * ((1 if i == 0 else hs) + hs + 2) for i in range(nl))
    params += 3 * hs * hs + 3 * hs + hs * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_attnlstm(1, hs, nl, nh, 10)
    desc = f"attnlstm-{hs}h-{nl}l-{nh}head"
    return desc, code, params, inp, outp, mt


def _sample_selfattn(lo: int, hi: int) -> Optional[Candidate]:
    dm = _rand_in_range(32, min(512, int(hi ** 0.4)), 512)
    nl = _rand_in_range(1, min(12, int(hi ** 0.2)), 20)
    nh = max(1, dm // 32)
    params = (3 * dm * dm + 3 * dm) * nl + dm * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_selfattn(dm, nl, nh, 10)
    desc = f"selfattn-{dm}d-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_gcn(lo: int, hi: int) -> Optional[Candidate]:
    in_feat = _rand_in_range(2, min(1024, int(hi ** 0.45)), 1024)
    hidden = _rand_in_range(4, min(1024, int(hi ** 0.4)), 1024)
    params = in_feat * hidden + hidden + hidden * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_gcn(in_feat, hidden, 10)
    desc = f"gcn-{in_feat}x{hidden}"
    return desc, code, params, inp, outp, mt


# ═══════════════════════════════════════════════════
#  Large-tier samplers (≥ 10M params)
# ═══════════════════════════════════════════════════

def _sample_vit(lo: int, hi: int) -> Optional[Candidate]:
    ps = random.choice([4, 8])
    dm = _rand_in_range(64, min(768, int(hi ** 0.38)), 768)
    nl = _rand_in_range(2, min(12, int(hi ** 0.15)), 16)
    nh = max(1, dm // 64)
    num_patches = (32 // ps) ** 2
    patch_dim = ps * ps * 3
    params = patch_dim * dm + dm + num_patches * dm
    layer_p = (4 * dm * dm + 4 * dm) + (dm * dm * 8 + dm * 4 + dm)
    params += layer_p * nl + dm * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_vit(ps, dm, nl, nh, 32, 10)
    desc = f"vit-p{ps}-d{dm}-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_unet(lo: int, hi: int) -> Optional[Candidate]:
    in_ch = random.choice([1, 3])
    base_ch = _rand_in_range(8, min(128, int(hi ** 0.3)), 128)
    ns = _rand_in_range(2, min(5, int(hi ** 0.1)), 6)
    params = 0
    prev_ch = in_ch
    ch = base_ch
    enc_chs = []
    for _ in range(ns):
        enc_chs.append(ch)
        params += prev_ch * ch * 9 + ch + ch * ch * 9 + ch
        prev_ch = ch
        ch *= 2
    params += prev_ch * ch * 9 + ch + ch * ch * 9 + ch
    prev_ch = ch
    for i in range(ns - 1, -1, -1):
        skip_ch = enc_chs[i]
        ch = skip_ch
        params += prev_ch * ch * 4 + ch
        params += (ch + skip_ch) * ch * 9 + ch + ch * ch * 9 + ch
        prev_ch = ch
    params += prev_ch * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_unet(in_ch, base_ch, ns, 10)
    desc = f"unet-{in_ch}ch-b{base_ch}-{ns}s"
    return desc, code, params, inp, outp, mt


def _sample_mixer(lo: int, hi: int) -> Optional[Candidate]:
    ps = random.choice([4, 8, 16])
    dm = _rand_in_range(64, min(512, int(hi ** 0.38)), 512)
    nl = _rand_in_range(1, min(8, int(hi ** 0.15)), 12)
    num_patches = (32 // ps) ** 2
    patch_dim = ps * ps * 3
    params = patch_dim * dm + dm
    token_mix_p = num_patches * num_patches * 2 + num_patches * 2
    channel_mix_p = dm * dm * 2 + dm * 2
    params += (token_mix_p + channel_mix_p) * nl + dm * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_mixer(ps, dm, nl, 32, 10)
    desc = f"mixer-p{ps}-d{dm}-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_gpt(lo: int, hi: int) -> Optional[Candidate]:
    vocab = _rand_in_range(100, min(10000, int(hi ** 0.3)), 10000)
    dm = _rand_in_range(64, min(1024, int(hi ** 0.38)), 1024)
    nl = _rand_in_range(2, min(12, int(hi ** 0.12)), 16)
    nh = max(1, dm // 64)
    bs = _rand_in_range(32, min(512, int(hi ** 0.2)), 512)
    params = vocab * dm + bs * dm
    layer_p = (4 * dm * dm + 4 * dm) + (dm * dm * 8 + dm * 4 + dm)
    params += layer_p * nl + dm * vocab + vocab
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_gpt(vocab, dm, nl, nh, bs)
    desc = f"gpt-v{vocab}-d{dm}-{nl}l"
    return desc, code, params, inp, outp, mt


def _sample_t5(lo: int, hi: int) -> Optional[Candidate]:
    vocab = _rand_in_range(100, min(10000, int(hi ** 0.3)), 10000)
    dm = _rand_in_range(64, min(768, int(hi ** 0.35)), 768)
    nl = _rand_in_range(1, min(8, int(hi ** 0.12)), 10)
    nh = max(1, dm // 64)
    params = vocab * dm * 2
    enc_layer = (4 * dm * dm + 4 * dm) + (dm * dm * 8 + dm * 4 + dm)
    dec_layer = enc_layer + (4 * dm * dm + 4 * dm)
    params += (enc_layer + dec_layer) * nl + dm * 10 + 10
    if not _ok(lo, hi, params):
        return None
    code, params, inp, outp, mt = make_t5(vocab, dm, nl, nh, 10)
    desc = f"t5-v{vocab}-d{dm}-{nl}l"
    return desc, code, params, inp, outp, mt


# ── Registry: maps architecture key to sampler ──

SAMPLERS: dict[str, Sampler] = {
    # Universal
    "unary":       _sample_unary,
    "linear":      _sample_linear,
    "mlp":         _sample_mlp,
    "deep":        _sample_deep,
    "wide":        _sample_wide,
    "resblock":    _sample_resblock,
    "highway":     _sample_highway,
    "moe":         _sample_moe,
    "transformer": _sample_transformer,
    "sae":         _sample_sae,
    # Limited-up
    "lstm":        _sample_lstm,
    "gru":         _sample_gru,
    "bilstm":      _sample_bilstm,
    "cnn":         _sample_cnn,
    "ae":          _sample_ae,
    "vae":         _sample_vae,
    "gan":         _sample_gan,
    "multitask":   _sample_multitask,
    "contrastive": _sample_contrastive,
    "siamese":     _sample_siamese,
    # Medium (≥ 100K)
    "rescnn":      _sample_rescnn,
    "sepcnn":      _sample_sepcnn,
    "densecnn":    _sample_densecnn,
    "attnlstm":    _sample_attnlstm,
    "selfattn":    _sample_selfattn,
    "gcn":         _sample_gcn,
    # Large (≥ 10M)
    "vit":         _sample_vit,
    "unet":        _sample_unet,
    "mixer":       _sample_mixer,
    "gpt":         _sample_gpt,
    "t5":          _sample_t5,
}

# Architecture pools for different range sizes
# Universal: work at any param count
_UNIVERSAL = ["mlp", "deep", "wide", "resblock", "highway", "moe", "transformer", "sae"]

# Small-only: only when hi < 200
_SMALL_ONLY = ["unary"]

# Limited-up: stop appearing above max_params
_LIMITED = {
    "linear":       5_000_000,
    "cnn":          5_000_000,
    "lstm":         5_000_000,
    "gru":          5_000_000,
    "bilstm":       5_000_000,
    "ae":           5_000_000,
    "vae":          5_000_000,
    "gan":         10_000_000,
    "multitask":   10_000_000,
    "contrastive": 10_000_000,
    "siamese":     10_000_000,
}

# Medium-tier: only when hi >= 100K
_MEDIUM = ["rescnn", "sepcnn", "densecnn", "attnlstm", "selfattn", "gcn"]

# Large-tier: only when hi >= 10M
_LARGE = ["vit", "unet", "mixer", "gpt", "t5"]


def _build_pool(hi: int, arch_filter: Optional[List[str]] = None) -> List[str]:
    """Build candidate pool based on parameter range upper bound."""
    if arch_filter:
        return [k for k in arch_filter if k in SAMPLERS]

    pool = list(_UNIVERSAL)

    if hi < 200:
        pool += _SMALL_ONLY

    for arch, max_p in _LIMITED.items():
        if hi <= max_p:
            pool.append(arch)

    if hi >= 100_000:
        pool += _MEDIUM

    if hi >= 10_000_000:
        pool += _LARGE

    return pool


def _apply_fixed_dims(candidate: Candidate, fixed_input: Optional[int],
                    fixed_output: Optional[int]) -> Optional[Candidate]:
    """Rewrite a candidate's code so its input/output dims match the
    requested fixed values, then recount parameters by instantiating it.
    Returns None if the rewrite cannot be instantiated."""
    desc, code, params, inp, outp, mtype = candidate
    new_in = fixed_input if fixed_input is not None else inp
    new_out = fixed_output if fixed_output is not None else outp
    if new_in == inp and new_out == outp:
        return candidate
    new_code = _rewrite_model_dims(code, inp, outp, new_in, new_out, mtype)
    # Keep the folder description honest about the rewritten input dim
    # (e.g. 'mlp-142x60x7' -> 'mlp-784x60x7' for MNIST).
    if new_in != inp:
        import re
        desc = re.sub(r'-\d+', f'-{new_in}', desc, count=1)
    try:
        ns = {}
        exec("import torch\nimport torch.nn as nn\n" + new_code.format(1), ns)
        new_params = count_params(ns["M1"]())
    except Exception:
        return None
    return desc, new_code, new_params, new_in, new_out, mtype


def find_candidates(lo: int, hi: int, count: int, seed: int = 42,
                    arch_filter: Optional[List[str]] = None,
                    fixed_input: Optional[int] = None,
                    fixed_output: Optional[int] = None,
                    max_attempts: int = 5000) -> List[Candidate]:
    """Find architecture candidates within [lo, hi] parameter range.

    Args:
        lo: Minimum parameter count.
        hi: Maximum parameter count.
        count: Target number of candidates.
        seed: Random seed for reproducibility.
        arch_filter: If provided, only sample from these architecture keys.
        fixed_input: If set, force all candidates to use this input dimension.
        fixed_output: If set, force all candidates to use this output dimension.
        max_attempts: Base sampling attempts, scaled by count if needed.

    Returns:
        List of Candidate tuples sorted by parameter count.
    """
    if lo < 0 or hi < 0:
        raise ValueError(f"Parameter range bounds must be non-negative (got {lo}-{hi}).")
    if lo >= hi:
        raise ValueError(f"Low bound ({lo}) must be less than high bound ({hi}).")

    random.seed(seed)
    results: List[Candidate] = []
    seen: set = set()

    pool = _build_pool(hi, arch_filter)
    if not pool:
        raise ValueError(f"No valid architectures. Available: {list(SAMPLERS.keys())}")

    # Scale attempts: ensure we try enough to fill the requested count.
    # Typical hit rate is ~10-30%, so multiply count by 10 as a safe margin.
    needed_attempts = max(max_attempts, count * 20)

    for _ in range(needed_attempts):
        if len(results) >= count:
            break

        arch_key = random.choice(pool)
        sampler = SAMPLERS.get(arch_key)
        if sampler is None:
            continue

        try:
            candidate = sampler(lo, hi)
            if candidate is not None:
                if fixed_input is not None or fixed_output is not None:
                    candidate = _apply_fixed_dims(candidate, fixed_input, fixed_output)
                    if candidate is None:
                        continue
                desc = candidate[0]
                if desc not in seen:
                    seen.add(desc)
                    results.append(candidate)
        except Exception:
            continue

    return results


def list_architectures() -> List[str]:
    """Return list of all available architecture keys."""
    return sorted(SAMPLERS.keys())
