from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = REPORT_DIR / "figures"


def _clean_label(value: str) -> str:
    return (
        str(value)
        .replace("_model", "")
        .replace("_test", "")
        .replace("_", " ")
        .replace(" on ", "\non ")
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_data_split() -> None:
    split = pd.read_csv(PROJECT_ROOT / "data" / "split_summary.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(split))
    bottom = np.zeros(len(split))
    colors = {"train": "#4C78A8", "validation": "#F58518", "test": "#54A24B"}
    for col in ["train", "validation", "test"]:
        ax.bar(x, split[col], bottom=bottom, label=col.title(), color=colors[col])
        bottom += split[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(split["dataset"].str.title())
    ax.set_ylabel("Rows")
    ax.set_title("Chronological 3:1:1 Data Split")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "data_split.png")


def plot_lgbm_performance() -> None:
    lgbm = pd.read_csv(PROJECT_ROOT / "LGBMmodel" / "model_comparison_test.csv", index_col=0)
    labels = [_clean_label(idx) for idx in lgbm.index]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    metrics = [
        ("Total Return", "Total Return (%)", 100.0, "#4C78A8"),
        ("Sharpe Ratio", "Sharpe Ratio", 1.0, "#F58518"),
        ("Max Drawdown", "Max Drawdown (%)", 100.0, "#E45756"),
    ]
    for ax, (col, ylabel, scale, color) in zip(axes, metrics, strict=True):
        values = lgbm[col].astype(float) * scale
        ax.bar(labels, values, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(col)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("LGBM Test Performance Across Model-Dataset Scenarios", y=1.03)
    _save(fig, "lgbm_test_performance.png")


def plot_grpo_risk_comparison() -> None:
    risk = pd.read_csv(PROJECT_ROOT / "outputs" / "risk_comparison.csv")
    labels = risk["method"].str.replace("_", " ").str.title()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    panels = [
        ("final_equity", "Final Equity (HKD mn)", 1e-6, "#4C78A8"),
        ("sharpe_ratio", "Sharpe Ratio", 1.0, "#F58518"),
        ("max_drawdown", "Max Drawdown (%)", 100.0, "#E45756"),
    ]
    for ax, (col, ylabel, scale, color) in zip(axes, panels, strict=True):
        ax.bar(labels, risk[col].astype(float) * scale, color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(col.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("GRPO Fixed Position vs Dynamic Risk Management", y=1.03)
    _save(fig, "grpo_risk_comparison.png")


def plot_garch_test_results() -> None:
    garch = pd.read_csv(
        PROJECT_ROOT
        / "submission_garch_bollinger"
        / "outputs"
        / "garch_bollinger_test_results.csv"
    )
    top = garch.sort_values("test_sharpe", ascending=False).head(12).copy()
    top["label"] = top["variant"].str.replace("_", " ") + "\n" + top["test_dataset"].str.replace("_test", "")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    axes[0].barh(top["label"], top["test_return"] * 100.0, color="#4C78A8")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Test Return (%)")
    axes[0].set_title("Top GARCH-Bollinger Test Returns")
    axes[0].grid(axis="x", alpha=0.25)
    axes[1].barh(top["label"], top["test_sharpe"], color="#F58518")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Test Sharpe")
    axes[1].set_title("Top GARCH-Bollinger Test Sharpe")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("GARCH-Bollinger Out-of-Sample Performance", y=1.02)
    _save(fig, "garch_test_results.png")


def plot_garch_validation_selection() -> None:
    grid = pd.read_csv(
        PROJECT_ROOT
        / "submission_garch_bollinger"
        / "outputs"
        / "garch_bollinger_validation_grid.csv"
    )
    nonzero = grid[grid["turnover"].astype(float) > 0].copy()
    grouped = (
        nonzero.groupby(["variant", "data_used_for_params"], as_index=False)
        .agg(validation_sharpe=("validation_sharpe", "max"), validation_return=("validation_return", "max"))
        .sort_values("validation_sharpe", ascending=False)
        .head(15)
    )
    grouped["label"] = grouped["variant"].str.replace("_", " ") + "\n" + grouped["data_used_for_params"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    scatter = ax.scatter(
        grouped["validation_return"] * 100.0,
        grouped["validation_sharpe"],
        s=120,
        c=np.arange(len(grouped)),
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
    )
    for _, row in grouped.iterrows():
        ax.annotate(row["label"], (row["validation_return"] * 100.0, row["validation_sharpe"]), fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Validation Return (%)")
    ax.set_ylabel("Validation Sharpe")
    ax.set_title("GARCH-Bollinger Validation Selection Landscape")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Rank by validation Sharpe")
    _save(fig, "garch_validation_selection.png")


def plot_dynamic_trade_pnl() -> None:
    ledger = pd.read_csv(PROJECT_ROOT / "outputs" / "dynamic_risk_trade_ledger.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    axes[0].hist(ledger["trade_pnl"], bins=60, color="#4C78A8", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Dynamic GRPO Trade PnL Distribution")
    axes[0].set_xlabel("Trade PnL (HKD)")
    axes[0].set_ylabel("Number of Trades")
    axes[0].grid(axis="y", alpha=0.25)
    ledger["balance_after_trade"].reset_index(drop=True).plot(ax=axes[1], color="#54A24B", linewidth=1.1)
    axes[1].set_title("Balance After Each Closed Trade")
    axes[1].set_xlabel("Trade Number")
    axes[1].set_ylabel("Balance (HKD)")
    axes[1].grid(alpha=0.25)
    _save(fig, "dynamic_trade_pnl.png")


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.axis("off")
    boxes = [
        ("Raw HI Futures\n1-min Bars", 0.06, 0.55),
        ("Chronological Split\nTrain / Validation / Test", 0.25, 0.55),
        ("LGBM\nReturn Forecast", 0.47, 0.78),
        ("GRPO\nPolicy Learning", 0.47, 0.55),
        ("GARCH-Bollinger\nVolatility Bands", 0.47, 0.32),
        ("Backtesting\nand Selection", 0.69, 0.55),
        ("Slippage and\nRisk Management", 0.86, 0.55),
    ]
    for text, x, y in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F7F7", edgecolor="#555555"),
        )
    arrows = [
        ((0.14, 0.55), (0.19, 0.55)),
        ((0.33, 0.58), (0.39, 0.76)),
        ((0.33, 0.55), (0.39, 0.55)),
        ((0.33, 0.52), (0.39, 0.34)),
        ((0.55, 0.78), (0.62, 0.60)),
        ((0.55, 0.55), (0.61, 0.55)),
        ((0.55, 0.32), (0.62, 0.50)),
        ((0.77, 0.55), (0.81, 0.55)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#333333"))
    ax.set_title("Project Research Pipeline", fontsize=14, pad=10)
    _save(fig, "project_workflow.png")


def write_summary_table() -> None:
    lgbm_risk = json.loads((PROJECT_ROOT / "LGBMmodel" / "risk_summary.json").read_text(encoding="utf-8"))
    grpo_risk = json.loads((PROJECT_ROOT / "outputs" / "dynamic_risk_summary.json").read_text(encoding="utf-8"))
    rows = [
        {
            "strategy": "LGBM risk-managed",
            "final_equity": lgbm_risk["final_capital"],
            "total_return": lgbm_risk["total_return"],
            "max_drawdown": lgbm_risk["max_drawdown"],
            "trades": lgbm_risk["total_trades"],
        },
        {
            "strategy": "GRPO dynamic risk",
            "final_equity": grpo_risk["metrics"]["final_equity"],
            "total_return": grpo_risk["metrics"]["final_equity"] / grpo_risk["initial_capital"] - 1.0,
            "max_drawdown": grpo_risk["metrics"]["max_drawdown"],
            "trades": grpo_risk["number_of_trades"],
        },
    ]
    pd.DataFrame(rows).to_csv(REPORT_DIR / "risk_summary_for_latex.csv", index=False)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_data_split()
    plot_lgbm_performance()
    plot_grpo_risk_comparison()
    plot_garch_test_results()
    plot_garch_validation_selection()
    plot_dynamic_trade_pnl()
    plot_workflow()
    write_summary_table()
    print(f"Generated figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
