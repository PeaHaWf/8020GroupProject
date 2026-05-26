from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR
from src.risk import run_risk_management


def main() -> None:
    summary = run_risk_management(OUTPUT_DIR)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
