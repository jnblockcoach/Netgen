"""Allow running netgen as `python -m netgen`.

Handles the case where the cwd is the netgen package directory itself,
which otherwise confuses Python's import system.
"""
import os
import sys

# When cwd is the netgen package directory, Python can't find 'netgen'
# because it's looking for netgen/netgen/. Fix by adding parent to path.
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
if os.getcwd() == _pkg_dir and _parent not in sys.path:
    sys.path.insert(0, _parent)

from .cli import main

main()
