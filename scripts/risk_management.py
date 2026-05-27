from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR, RiskConfig
from src.risk import run_risk_management


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic risk management for the selected best model.")
    parser.add_argument("--risk-per-trade", type=float, default=RiskConfig.risk_per_trade)
    parser.add_argument("--max-contracts", type=int, default=RiskConfig.max_contracts)
    parser.add_argument("--protective-drawdown", type=float, default=RiskConfig.protective_drawdown)
    parser.add_argument("--kill-switch-drawdown", type=float, default=RiskConfig.kill_switch_drawdown)
    parser.add_argument("--cooldown-bars", type=int, default=RiskConfig.cooldown_bars)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    risk_config = RiskConfig(
        risk_per_trade=args.risk_per_trade,
        max_contracts=args.max_contracts,
        protective_drawdown=args.protective_drawdown,
        kill_switch_drawdown=args.kill_switch_drawdown,
        cooldown_bars=args.cooldown_bars,
    )
    summary = run_risk_management(OUTPUT_DIR, risk_config=risk_config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
