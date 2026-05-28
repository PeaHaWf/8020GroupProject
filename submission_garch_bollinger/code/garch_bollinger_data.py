"""Data loading and bar preparation for the GARCH-Bollinger experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "hi1_open",
    "hi1_high",
    "hi1_low",
    "hi1_close",
    "hi1_volume",
}


def load_processed_split(data_dir: Path, dataset: str, split: str) -> pd.DataFrame:
    path = data_dir / f"{dataset}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split file: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    return df.sort_values("timestamp").reset_index(drop=True)

def trading_date_from_timestamp(timestamp: pd.Series) -> pd.Series:
    adjusted = timestamp.where(timestamp.dt.hour.gt(3), timestamp - pd.Timedelta(days=1))
    return adjusted.dt.date.astype(str)

def add_segments(df: pd.DataFrame, gap_minutes: int = 2) -> pd.DataFrame:
    out = df.copy()
    out["trading_date"] = trading_date_from_timestamp(out["timestamp"])
    gaps = out["timestamp"].diff().dt.total_seconds().div(60).fillna(0.0)
    out["segment_start"] = gaps.gt(gap_minutes)
    if not out.empty:
        out.loc[out.index[0], "segment_start"] = True
    out["segment_id"] = out["segment_start"].cumsum()
    return out

def resample_inside_segments(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    if bar_minutes < 1:
        raise ValueError("bar_minutes must be >= 1")

    segmented = add_segments(df)
    if bar_minutes == 1:
        out = segmented.copy()
        out["bar_minutes"] = 1
        out["source_rows"] = 1
        return out

    rows: list[dict[str, object]] = []
    for _, group in segmented.groupby("segment_id", sort=False):
        group = group.reset_index(drop=True)
        group["bar_id"] = np.arange(len(group)) // bar_minutes
        aggregated = group.groupby("bar_id", sort=False).agg(
            timestamp=("timestamp", "last"),
            trading_date=("trading_date", "first"),
            hi1_open=("hi1_open", "first"),
            hi1_high=("hi1_high", "max"),
            hi1_low=("hi1_low", "min"),
            hi1_close=("hi1_close", "last"),
            hi1_volume=("hi1_volume", "sum"),
            segment_id=("segment_id", "first"),
            source_rows=("timestamp", "size"),
        )
        rows.extend(aggregated.to_dict("records"))

    out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    out["bar_minutes"] = bar_minutes
    out["segment_start"] = out["segment_id"].ne(out["segment_id"].shift(1))
    return out

def add_close_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close_return = out.groupby("segment_id", sort=False)["hi1_close"].pct_change()
    out["close_return"] = close_return.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out.loc[out["segment_start"].astype(bool), "close_return"] = 0.0
    return out

def prepare_bars(
    frames: dict[str, pd.DataFrame],
    bar_minutes_values: tuple[int, ...],
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
) -> dict[tuple[str, str, int], pd.DataFrame]:
    bars: dict[tuple[str, str, int], pd.DataFrame] = {}
    for dataset in datasets:
        for split in splits:
            frame = frames[f"{dataset}_{split}"]
            for bar_minutes in bar_minutes_values:
                bars[(dataset, split, bar_minutes)] = add_close_returns(
                    resample_inside_segments(frame, bar_minutes)
                )

    for dataset in datasets:
        for bar_minutes in bar_minutes_values:
            train_val = pd.concat(
                [frames[f"{dataset}_train"], frames[f"{dataset}_validation"]],
                ignore_index=True,
            ).sort_values("timestamp").reset_index(drop=True)
            bars[(dataset, "train_validation", bar_minutes)] = add_close_returns(
                resample_inside_segments(train_val, bar_minutes)
            )
    return bars

def write_data_summary(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for split in splits:
            frame = frames[f"{dataset}_{split}"]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "rows": len(frame),
                    "start": frame["timestamp"].min(),
                    "end": frame["timestamp"].max(),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "data_used_summary.csv", index=False)
    return summary

