"""NetGen - Batch Neural Network Model Generator.

Generate diverse PyTorch model training folders with verified parameter counts.

Usage:
    python -m netgen --range 10000-20000 --count 20
    python -m netgen --range 10000-20000 --count 20 --arch mlp,lstm
"""

from .architectures import (
    make_linear, make_mlp,
    make_unary_a, make_unary_b, make_unary_c,
    make_lstm, make_gru, make_bilstm,
    make_cnn, make_ae, make_vae,
    make_deep_mlp, make_stacked_ae,
    make_transformer, make_wide_net,
    make_resblock, make_highway, make_moe,
    make_multitask, make_gan,
    make_contrastive, make_siamese,
    count_params,
)
from .search import find_candidates, list_architectures
from .generator import gen_folder
from .templates import get_templates
from .datasets import get_dataset_code, list_datasets

__all__ = [
    # Architecture generators
    'make_linear', 'make_mlp',
    'make_unary_a', 'make_unary_b', 'make_unary_c',
    'make_lstm', 'make_gru', 'make_bilstm',
    'make_cnn', 'make_ae', 'make_vae',
    'make_deep_mlp', 'make_stacked_ae',
    'make_transformer', 'make_wide_net',
    'make_resblock', 'make_highway', 'make_moe',
    'make_multitask', 'make_gan',
    'make_contrastive', 'make_siamese',
    'count_params',
    # Search
    'find_candidates', 'list_architectures',
    # Generator
    'gen_folder',
    # Templates
    'get_templates',
    # Datasets
    'get_dataset_code', 'list_datasets',
]
