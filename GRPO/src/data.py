from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_NAMES, SPLIT_NAMES, dataset_path


PRICE_COLUMNS = ["hi1_open", "hi1_high", "hi1_low", "hi1_close"]
VOLUME_COLUMN = "hi1_volume"


def _time_to_hhmmss(value: object) -> str:
    return str(int(value)).zfill(6)


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"date", "time", *PRICE_COLUMNS, VOLUME_COLUMN}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")

    date_part = df["date"].astype(str)
    time_part = df["time"].map(_time_to_hhmmss)
    df["timestamp"] = pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M%S")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def split_day_night(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    minutes = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    day_mask = ((minutes >= 9 * 60 + 15) & (minutes <= 12 * 60)) | (
        (minutes >= 13 * 60) & (minutes <= 16 * 60 + 30)
    )
    night_mask = (minutes >= 17 * 60 + 15) | (minutes <= 3 * 60)

    day = df.loc[day_mask].copy().reset_index(drop=True)
    night = df.loc[night_mask].copy().reset_index(drop=True)
    return day, night


def chronological_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if len(df) < 5:
        raise ValueError("need at least 5 rows to create a 3:1:1 split")

    train_end = int(np.floor(len(df) * 0.6))
    validation_end = int(np.floor(len(df) * 0.8))
    return {
        "train": df.iloc[:train_end].copy().reset_index(drop=True),
        "validation": df.iloc[train_end:validation_end].copy().reset_index(drop=True),
        "test": df.iloc[validation_end:].copy().reset_index(drop=True),
    }


def write_dataset_with_splits(name: str, df: pd.DataFrame, data_dir: Path) -> dict[str, int]:
    full_path = dataset_path(name)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)

    counts = {name: len(df)}
    for split, split_df in chronological_splits(df).items():
        out_path = dataset_path(name, split)
        split_df.to_csv(out_path, index=False)
        counts[f"{name}_{split}"] = len(split_df)
    return counts


def prepare_datasets(raw_path: Path, data_dir: Path) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    original = load_raw_data(raw_path)
    day, night = split_day_night(original)

    summary_rows = []
    for name, frame in zip(DATASET_NAMES, (original, day, night), strict=True):
        counts = write_dataset_with_splits(name, frame, data_dir)
        summary_rows.append(
            {
                "dataset": name,
                "rows": counts[name],
                **{split: counts[f"{name}_{split}"] for split in SPLIT_NAMES},
                "start": frame["timestamp"].min(),
                "end": frame["timestamp"].max(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(data_dir / "split_summary.csv", index=False)
    return summary
