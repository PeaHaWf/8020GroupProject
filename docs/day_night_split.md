# 日盘与夜盘划分方法

## 数据来源

原始数据文件为 `原始数据/hi1_20170701_20200609.csv`，字段包括 `date`、`time`、`hi1_open`、`hi1_high`、`hi1_low`、`hi1_close`、`hi1_volume`。

## 时间戳处理

`date` 使用 `YYYYMMDD` 格式，`time` 使用 `HHMMSS` 格式。脚本会先将两列合并为 `timestamp`，再按时间顺序排序。

## 划分规则

日盘：

- 09:15 至 12:00
- 13:00 至 16:30

夜盘：

- 17:15 至次日 03:00

该规则在 `GRPO/src/data.py` 的 `split_day_night()` 中定义。如需修改交易时段，只需要调整该函数中的分钟数边界。

## 输出文件

运行：

```bash
python GRPO/scripts/prepare_data.py
```

会生成：

- `data/original_data.csv`
- `data/day_data.csv`
- `data/night_data.csv`
- `data/*_train.csv`
- `data/*_validation.csv`
- `data/*_test.csv`
- `data/split_summary.csv`

所有 train/validation/test 均按时间顺序以 3:1:1 划分。
