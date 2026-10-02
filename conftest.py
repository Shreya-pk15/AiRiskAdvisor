"""
Pytest configuration file.
Adds project root directory to sys.path to ensure module imports resolve correctly during test execution.
"""
import sys
import os
import numpy as np

# NumPy 2.0 compatibility monkey-patch for chromadb
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint"):
    np.uint = np.uint64

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
