"""
Pytest configuration file.
Adds project root directory to sys.path to ensure module imports resolve correctly during test execution.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
