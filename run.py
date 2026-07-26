"""Quick launcher — run netgen from the project directory.

Usage:
    python run.py --range 10000-20000 --count 20
"""
import sys
import os

# The project dir *is* the package dir, so add it to sys.path
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# Now we can import netgen modules directly
from cli import run

sys.exit(run())
