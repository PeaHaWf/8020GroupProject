from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR
from src.evaluation import evaluate_combinations


def main() -> None:
    results = evaluate_combinations(OUTPUT_DIR)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
