# GARCH-Bollinger Submission Files

This folder contains the GARCH-Bollinger traditional strategy line.

## Report Drafts

- `project_guideline_sections_2_7.md`: project guideline sections 2-7 for the GARCH-Bollinger strategy.
- `garch_bollinger_first4_report.md`: earlier first-four-section generated report.

## Code

Run from the project root:

```bash
python code/garch_bollinger_first4_from_processed_data.py
```

Code files:

- `code/garch_bollinger_first4_from_processed_data.py`: main entrypoint, report/output orchestration.
- `code/garch_bollinger_data.py`: processed CSV loading, segmenting, resampling, returns.
- `code/garch_bollinger_strategy.py`: GARCH, Bollinger bands, volume confirmation, signal generation.
- `code/garch_bollinger_backtest.py`: non-capital backtest, metrics, validation selection, test evaluation.

## Key Outputs

- `outputs/data_used_summary.csv`: train/validation/test rows and date ranges.
- `outputs/garch_model_params.csv`: fitted GARCH parameters.
- `outputs/garch_bollinger_validation_selected_results.csv`: selected validation parameters and validation metrics.
- `outputs/garch_bollinger_selected_params.csv`: compact selected parameter table.
- `outputs/garch_bollinger_test_results.csv`: final out-of-sample test results.

The backtest uses ratio returns only. It does not use initial capital, final equity, or wealth.
