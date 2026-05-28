from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, RAW_DATA_PATH
from src.data import prepare_datasets


def main() -> None:
    summary = prepare_datasets(RAW_DATA_PATH, DATA_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
