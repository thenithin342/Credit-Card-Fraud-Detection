"""src/evaluation/evaluate.py
────────────────────────────────────────────────────────────────────────
CLI entry point alias for evaluating trained ML models.
Delegates directly to `src.training.evaluate.main`.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys

from src.training.evaluate import main

if __name__ == "__main__":
    sys.exit(main() or 0)
