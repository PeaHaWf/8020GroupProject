# Day/Night Session Split

## Data Source

Raw data file: `data/hi1_20170701_20200609.csv`. Columns: `date`, `time`, `hi1_open`, `hi1_high`, `hi1_low`, `hi1_close`, `hi1_volume`.

## Timestamp Handling

`date` uses `YYYYMMDD` format, `time` uses `HHMMSS` format. The script concatenates both into a `timestamp` column and sorts chronologically.

## Split Rules

Day session:

- 09:15 to 12:00
- 13:00 to 16:30

Night session:

- 17:15 to 03:00 (next day)

Defined in `GRPO/src/data.py:split_day_night()`. To change session boundaries, adjust the minute thresholds in that function.

## Output Files

Run:

```bash
python GRPO/scripts/prepare_data.py
```

Generates:

- `data/original_data.csv`
- `data/day_data.csv`
- `data/night_data.csv`
- `data/*_train.csv`
- `data/*_validation.csv`
- `data/*_test.csv`
- `data/split_summary.csv`

All train/validation/test splits are chronological with a 3:1:1 ratio.
