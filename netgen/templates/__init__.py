"""Template dispatcher — routes to the correct tier based on parameter count.

Usage:
    from .templates import get_templates
    data, train, eval = get_templates(tier, model_type)
"""

def get_templates(tier: str, model_type: str):
    """Return (data_code, train_code, eval_code) for a given tier and model type.

    Args:
        tier: 'quick' (<50K params), 'standard' (50K~50M), 'production' (>50M)
        model_type: 'ce', 'mse', 'cnn', 'rnn', 'ae', 'vae', 'mt', 'gan',
                    'contrastive', 'siamese', 'gcn'
    """
    if tier == 'quick':
        from . import quick
        return quick.get_templates(model_type)
    elif tier == 'standard':
        from . import standard
        return standard.get_templates(model_type)
    elif tier == 'production':
        from . import production
        return production.get_templates(model_type)
    else:
        raise ValueError(f"Unknown tier: {tier}")


def get_extra_files(tier: str, model_type: str, class_name: str):
    """Return dict of {filename: content} for tier-specific extra files.

    Quick tier: no extras (all handled by generator.py directly)
    Standard tier: sweep.py, visualize.py (real), predict.py
    Production tier: all standard extras + benchmark.py, profile.py, export.py
    """
    if tier == 'standard':
        from . import standard
        return standard.get_extra_files(model_type, class_name)
    elif tier == 'production':
        from . import production
        return production.get_extra_files(model_type, class_name)
    else:
        return {}
