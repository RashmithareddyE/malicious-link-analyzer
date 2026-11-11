"""Make the project root importable and keep tests offline-friendly."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MLA_EMBEDDER", "tfidf")   # no model download during tests
