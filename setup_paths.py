# setup_paths.py
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.getcwd() != BASE_DIR:
    os.chdir(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"CWD set to {BASE_DIR}")
