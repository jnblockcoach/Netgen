"""Shared pytest fixtures: make the netgen package importable and provide
a per-test scratch directory for generated models.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest


@pytest.fixture()
def scratch(tmp_path):
    """A temp directory used as the output dir for generated models."""
    return str(tmp_path)
