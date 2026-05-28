

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================================
# global cfg
# ============================================================
DATA_DIR = '../data/'
TRANSACTION_COST_RATE = 0.0001   # single side cost
SLIPPAGE_POINTS = 1              # 1 index point
HSI_MULTIPLIER = 50              # 50 per point
INITIAL_CAPITAL = 5_000_000
N_FOLDS = 5                      # K-fold cross validation

# ============================================================
# 1. data loading
# ============================================================

def load_and_merge_data(dataset_type):

    train_path = os.path.join(DATA_DIR, f'{dataset_type}_train.csv')
    val_path = os.path.join(DATA_DIR, f'{dataset_type}_validation.csv')
    test_path = os.path.join(DATA_DIR, f'{dataset_type}_test.csv')

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # merge train + val
    merged = pd.concat([train, val], ignore_index=True)

    merged['dataset'] = 'train_val'
    test['dataset'] = 'test'

    for df in [merged, test]:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"[{dataset_type}] train+val: {len(merged)}, test: {len(test)}")
    return merged, test

def load_validation_data(dataset_type):
    """load validation set"""
    val_path = os.path.join(DATA_DIR, f'{dataset_type}_validation.csv')
    val = pd.read_csv(val_path)
    val['datetime'] = pd.to_datetime(val['timestamp'])
    val.sort_values('datetime', inplace=True)
    val.reset_index(drop=True, inplace=True)
    val['dataset'] = 'validation'
    print(f"[{dataset_type}] validation: {len(val)} 条")
    return val

def load_train_data(dataset_type):
    """load train set"""
    train_path = os.path.join(DATA_DIR, f'{dataset_type}_train.csv')
    train = pd.read_csv(train_path)
    train['datetime'] = pd.to_datetime(train['timestamp'])
    train.sort_values('datetime', inplace=True)
    train.reset_index(drop=True, inplace=True)
    train['dataset'] = 'train'
    print(f"[{dataset_type}] train: {len(train)} 条")
    return train
# ============================================================
# 2. feature engineer
# ============================================================

def compute_features(df):

    df = df.copy()

    o = df['hi1_open'].values.astype(float)
    h = df['hi1_high'].values.astype(float)
    l = df['hi1_low'].values.astype(float)
    c = df['hi1_close'].values.astype(float)
    v = df['hi1_volume'].values.astype(float)

    df['ret_1'] = np.nan
    df['ret_1'].values[1:] = (c[1:] - c[:-1]) / c[:-1]

    df['log_ret_1'] = np.nan
    df['log_ret_1'].values[1:] = np.log(c[1:] / c[:-1])
    
    # lag-return
    for lag in [2, 3, 5, 10, 15, 30]:
        ret = np.full(len(df), np.nan)
        ret[lag:] = (c[lag:] - c[:-lag]) / c[:-lag]
        df[f'ret_{lag}'] = ret
    
    # volatility
    for window in [5, 10, 15, 30]:
        df[f'vol_{window}'] = df['ret_1'].rolling(window).std()
    
    # volume
    df['log_volume'] = np.log(v + 1)
    df['vol_ma_5'] = df['hi1_volume'].rolling(5).mean()
    df['vol_ma_10'] = df['hi1_volume'].rolling(10).mean()
    df['vol_ratio_5'] = df['hi1_volume'] / (df['vol_ma_5'] + 1e-8)
    df['vol_ratio_10'] = df['hi1_volume'] / (df['vol_ma_10'] + 1e-8)
    
    # --- price ---
    df['high_low_spread'] = (h - l) / (c + 1e-8)
    df['close_open_ratio'] = (c - o) / (o + 1e-8)
    df['upper_shadow'] = (h - np.maximum(o, c)) / (c + 1e-8)
    df['lower_shadow'] = (np.minimum(o, c) - l) / (c + 1e-8)
    

    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = df['hi1_close'].rolling(window).mean()
    

    for window in [5, 10, 20, 60]:
        df[f'ma_dev_{window}'] = (df['hi1_close'] - df[f'ma_{window}']) / (df[f'ma_{window}'] + 1e-8)

    df['ma_cross_5_20'] = df['ma_5'] / (df['ma_20'] + 1e-8) - 1
    df['ma_cross_10_60'] = df['ma_10'] / (df['ma_60'] + 1e-8) - 1

    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    for window in [6, 14, 24]:
        avg_gain = pd.Series(gain).rolling(window).mean().values
        avg_loss = pd.Series(loss).rolling(window).mean().values
        rs = avg_gain / (avg_loss + 1e-8)
        df[f'rsi_{window}'] = 100 - 100 / (1 + rs)

    df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Mon
    
    # time encoding
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / (24 * 60))
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / (24 * 60))
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    # label
    df['target'] = np.nan
    df['target'].values[:-1] = (c[1:] - c[:-1]) / c[:-1]
    
    # up or down
    df['target_direction'] = (df['target'] > 0).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def get_feature_columns(df):

    exclude = ['date', 'time', 'timestamp', 'datetime', 'dataset',
               'target', 'target_direction',
               'hi1_open', 'hi1_high', 'hi1_low', 'hi1_close', 'hi1_volume']
    return [col for col in df.columns if col not in exclude]


# ============================================================
# 3. K-fold hyperparam selection
# ============================================================

def hyperparameter_tuning(X, y, dataset_name):

    param_grid = [
        {'n_estimators': 200, 'max_depth': 6, 'num_leaves': 63, 'learning_rate': 0.05,
         'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'max_depth': 8, 'num_leaves': 127, 'learning_rate': 0.03,
         'min_child_samples': 30, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 500, 'max_depth': 5, 'num_leaves': 31, 'learning_rate': 0.02,
         'min_child_samples': 100, 'subsample': 0.9, 'colsample_bytree': 0.9},
        {'n_estimators': 400, 'max_depth': 7, 'num_leaves': 95, 'learning_rate': 0.04,
         'min_child_samples': 60, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 300, 'max_depth': 6, 'num_leaves': 63, 'learning_rate': 0.03,
         'min_child_samples': 40, 'subsample': 0.85, 'colsample_bytree': 0.85},
    ]
    
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    
    best_score = np.inf
    best_params = param_grid[0]
    
    print(f"\n  [{dataset_name}] 超参数搜索 (TimeSeriesSplit, {N_FOLDS}-fold)")
    print(f"  {'Param Set':<8} {'Fold RMSE':<60} {'Mean RMSE'}")
    print(f"  {'-'*8} {'-'*60} {'-'*12}")
    # grid search
    for i, params in enumerate(param_grid):
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                boosting_type='gbdt',
                verbosity=-1,
                random_state=42,
                n_jobs=-1,
                **params
            )
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            fold_scores.append(rmse)
        
        mean_rmse = np.mean(fold_scores)
        scores_str = ' | '.join([f'{s:.6f}' for s in fold_scores])
        print(f"  Set {i+1:<4} {scores_str:<60} {mean_rmse:.6f}")
        
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params
    
    print(f"  => 最佳参数: {best_params}, CV RMSE={best_score:.6f}")
    return best_params, best_score


# ============================================================
# 4. model training
# ============================================================

def train_model(X_train, y_train, params, dataset_name):

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
        **params
    )
    model.fit(X_train, y_train)
    print(f"  [{dataset_name}] model training completed")
    return model


# ============================================================
# 5. backtest(without risk management)
# ============================================================

def simulate_trading(predictions, df_test, transaction_cost_rate=TRANSACTION_COST_RATE,
                     slippage_points=SLIPPAGE_POINTS, multiplier=HSI_MULTIPLIER):

    actual_returns = df_test['target'].values
    close_prices = df_test['hi1_close'].values
    

    cost_threshold = transaction_cost_rate * 2
    
    positions = np.zeros(len(predictions))
    for i in range(len(predictions)):
        slippage_ret = slippage_points / (close_prices[i] + 1e-8)
        total_cost = cost_threshold + slippage_ret * 2
        if predictions[i] > total_cost:
            positions[i] = 1
        elif predictions[i] < -total_cost:
            positions[i] = -1
        else:
            positions[i] = 0
    
    # return
    strategy_returns = np.zeros(len(predictions))
    prev_position = 0
    trade_log = []
    trade_id = 0
    
    for i in range(len(predictions)):
        gross_ret = positions[i] * actual_returns[i]
        
        # transaction cost
        turnover_cost = 0
        if positions[i] != prev_position:

            if prev_position != 0:
                turnover_cost += transaction_cost_rate  # close a position
                slippage_cost = slippage_points / (close_prices[i] + 1e-8)
                turnover_cost += slippage_cost
            if positions[i] != 0:
                turnover_cost += transaction_cost_rate  # open
                slippage_cost = slippage_points / (close_prices[i] + 1e-8)
                turnover_cost += slippage_cost
            

            trade_id += 1
            trade_log.append({
                'trade_id': trade_id,
                'entry_time': df_test['datetime'].iloc[i],
                'entry_price': close_prices[i],
                'position': positions[i],
                'turnover_cost': turnover_cost,
            })
        
        net_ret = gross_ret - turnover_cost
        strategy_returns[i] = net_ret
        prev_position = positions[i]

    if prev_position != 0:
        last_cost = transaction_cost_rate
        last_cost += slippage_points / (close_prices[-1] + 1e-8)
        strategy_returns[-1] -= last_cost

    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
    return positions, strategy_returns, cumulative_returns, trade_log


def calculate_metrics(strategy_returns, positions, df_test):
    n = len(strategy_returns)

    # ==================== annualized factor ====================
    if 'datetime' in df_test.columns and len(df_test) > 0:
        df_date = df_test.copy()
        df_date['date'] = pd.to_datetime(df_date['datetime']).dt.date
        bars_per_day = df_date.groupby('date').size()
        avg_bars_per_day = bars_per_day.mean()
        num_days = len(bars_per_day)
        print(f"  [Dynamic Annualization] Avg bars/day: {avg_bars_per_day:.2f} "
              f"(based on {num_days} days)")
    else:
        avg_bars_per_day = 330.0
        print("  [Warning] default 330")

    trading_days_per_year = 252
    periods_per_year = avg_bars_per_day * trading_days_per_year
    # ========================================================

    # Log Return
    log_return = np.sum(np.log(1 + np.clip(strategy_returns, -0.99, None)))
    total_return = np.exp(log_return) - 1

    # Annualized Return
    annual_return = (1 + total_return) ** (periods_per_year / n) - 1 if n > 0 else 0.0

    # Sharpe Ratio (年化)
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    sharpe_ratio = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    # Max Drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

    # Win Rate
    trades = positions != 0
    win_rate = np.mean(strategy_returns[trades] > 0) if np.sum(trades) > 0 else 0.0

    # Profit Factor
    positive_rets = strategy_returns[trades][strategy_returns[trades] > 0]
    negative_rets = strategy_returns[trades][strategy_returns[trades] < 0]
    profit_factor = (np.sum(positive_rets) / np.sum(np.abs(negative_rets))) if len(negative_rets) > 0 else (
        np.inf if len(positive_rets) > 0 else 0.0)

    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Average trade return
    avg_trade_return = np.mean(strategy_returns[trades]) if np.sum(trades) > 0 else 0.0

    metrics = {
        'Log Return': log_return,
        'Total Return': total_return,
        'Annualized Return': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Calmar Ratio': calmar_ratio,
        'Avg Trade Return': avg_trade_return,
        'Total Trades': trade_log_count(positions),
        'Avg Bars Per Day': float(avg_bars_per_day),  # 新增，方便查看
    }
    return metrics


def trade_log_count(positions):

    changes = np.diff(positions, prepend=0)
    return np.sum(changes != 0)


# ============================================================
# 新增：用 validation set 进行模型选择
# ============================================================
def evaluate_on_validation(models, datasets):

    val_results = {}
    print(f"\n{'=' * 80}")
    print(" Validation Set model selection")
    print(f"{'=' * 80}")

    scenarios = [
        ('original_model on original_val', models['original'], 'original'),
        ('original_model on day_val', models['original'], 'day'),
        ('original_model on night_val', models['original'], 'night'),
        ('day_model on day_val', models['day'], 'day'),
        ('night_model on night_val', models['night'], 'night'),
    ]

    for name, model, ds_type in scenarios:
        val_data = load_validation_data(ds_type)

        val_feat = compute_features(val_data)
        X_val = val_feat[get_feature_columns(val_feat)]
        preds, pos, srets, cumrets, tlog, metrics = evaluate_on_test(model, X_val, val_feat, name)
        val_results[name] = {'metrics': metrics,
                             'score': metrics['Sharpe Ratio'] * 0.5 + metrics['Calmar Ratio'] * 0.3 + metrics[
                                 'Profit Factor'] * 0.2}


    best_name = max(val_results, key=lambda k: val_results[k]['score'])
    print(f"\n ===> Validation best model: {best_name}")
    return best_name, val_results
def evaluate_on_test(model, X_test, df_test, scenario_name):

    predictions = model.predict(X_test)
    positions, strategy_rets, cum_rets, trade_log = simulate_trading(predictions, df_test)


    metrics = calculate_metrics(strategy_rets, positions, df_test)

    print(f"\n [{scenario_name}]")
    print(f" Log Return: {metrics['Log Return']:.4f}")
    print(f" Total Return: {metrics['Total Return']:.4%}")
    print(f" Annualized Return: {metrics['Annualized Return']:.4%}")
    print(f" Sharpe Ratio: {metrics['Sharpe Ratio']:.4f}")
    print(f" Max Drawdown: {metrics['Max Drawdown']:.4%}")
    print(f" Win Rate: {metrics['Win Rate']:.4%}")
    print(f" Profit Factor: {metrics['Profit Factor']:.4f}")
    print(f" Calmar Ratio: {metrics['Calmar Ratio']:.4f}")
    print(f" Total Trades: {metrics['Total Trades']}")
    print(f" Avg Bars/Day: {metrics['Avg Bars Per Day']:.1f}")

    return predictions, positions, strategy_rets, cum_rets, trade_log, metrics


# ============================================================
# 6. main
# ============================================================

def main():
    print("=" * 80)
    print("  LGBM model train and evaluate")
    print("=" * 80)


    all_models = {}
    all_params = {}
    all_features = {}
    all_val_data = {}
    all_test_data = {}
    all_test_features = {}


    for ds in ['original', 'day', 'night']:
        print(f"\n{'='*60}")
        print(f" 处理 {ds.upper()} 数据集")
        print(f"{'='*60}")


        train = load_train_data(ds)
        val = load_validation_data(ds)
        test = load_and_merge_data(ds)[1]


        print(f" feature engineering...")
        train_feat = compute_features(train)
        val_feat = compute_features(val)
        test_feat = compute_features(test)

        feature_cols = get_feature_columns(train_feat)
        all_features[ds] = feature_cols

        X_train = train_feat[feature_cols]
        y_train = train_feat['target']

        all_val_data[ds] = val_feat
        all_test_data[ds] = test_feat
        all_test_features[ds] = test_feat[feature_cols]

        print(f" features: {len(feature_cols)}, "
              f"train: {len(X_train)}, "
              f"val: {len(val_feat)}, "
              f"test: {len(test_feat)}")


        best_params, cv_score = hyperparameter_tuning(X_train, y_train, ds)
        all_params[ds] = best_params

        # 4. 仅使用 train 数据训练最终模型
        print(f"\n training final model")
        model = train_model(X_train, y_train, best_params, ds)
        all_models[ds] = model


    print(f"\n{'='*80}")
    print(" best scenario:")
    print(f"{'='*80}")

    best_model_name, val_results = evaluate_on_validation(all_models, all_val_data)

    print(f"\n{'='*80}")
    print(" Test Set eval")
    print(f"{'='*80}")

    results = {}

    scenarios = [
        ('original_model on original_test', all_models['original'], all_test_features['original'], all_test_data['original']),
        ('original_model on day_test', all_models['original'], all_test_features['day'], all_test_data['day']),
        ('original_model on night_test', all_models['original'], all_test_features['night'], all_test_data['night']),
        ('day_model on day_test', all_models['day'], all_test_features['day'], all_test_data['day']),
        ('night_model on night_test', all_models['night'], all_test_features['night'], all_test_data['night']),
    ]

    for name, model, X_t, df_t in scenarios:
        preds, pos, srets, cumrets, tlog, metrics = evaluate_on_test(model, X_t, df_t, name)
        results[name] = {
            'predictions': preds,
            'positions': pos,
            'strategy_returns': srets,
            'cumulative_returns': cumrets,
            'trade_log': tlog,
            'metrics': metrics,
        }


    print(f"\n{'='*80}")
    print(" Test Set comparison")
    print(f"{'='*80}")

    comparison = pd.DataFrame({
        name: {
            'Log Return': res['metrics']['Log Return'],
            'Total Return': res['metrics']['Total Return'],
            'Annualized Return': res['metrics']['Annualized Return'],
            'Sharpe Ratio': res['metrics']['Sharpe Ratio'],
            'Max Drawdown': res['metrics']['Max Drawdown'],
            'Win Rate': res['metrics']['Win Rate'],
            'Profit Factor': res['metrics']['Profit Factor'],
            'Calmar Ratio': res['metrics']['Calmar Ratio'],
            'Total Trades': res['metrics']['Total Trades'],
        } for name, res in results.items()
    }).T

    print(comparison.to_string())
    comparison.to_csv('model_comparison_test.csv', index=True)
    print("\n Test result saved to model_comparison_test.csv")


    if 'original' in best_model_name:
        best_model = all_models['original']
    elif 'day' in best_model_name:
        best_model = all_models['day']
    else:
        best_model = all_models['night']

    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\n ===> Validation best model: {best_model_name}")
    print(f" best model saved to best_model.pkl")


    with open('best_model_info.json', 'w') as f:
        json.dump({
            'best_model': best_model_name,
            'val_scores': val_results
        }, f, indent=2, default=str)

    return all_models, results, best_model_name


if __name__ == '__main__':
    all_models, results, best_model_name = main()
