"""src/features/build_features.py
────────────────────────────────────────────────────────────────────────
CLI entry point alias for building offline features store.
Delegates directly to `src.features.offline_store.main`.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys

from src.features.offline_store import main

if __name__ == "__main__":
    sys.exit(main() or 0)
