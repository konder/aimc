"""
STEVE-1 Package Initialization
Adds the parent directory to sys.path for proper imports
"""
import os
import sys

# Get the absolute path of the parent directory (src/training)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

# Add to sys.path if not already there
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Project root and data directory
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '../../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

