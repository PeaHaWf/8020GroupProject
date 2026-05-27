"""
模型训练及评估脚本
========================
功能：
1. 数据加载与合并（train + val）
2. 特征工程
3. K-fold (TimeSeriesSplit) 交叉验证 + 超参数筛选
4. 训练三个LGBM模型（original / day / night）
5. 测试集评估（5种场景），含 transaction cost 和 slippage
6. 指标：Max Drawdown, Sharpe Ratio, Log Return 等
7. 比较分析，选出最佳模型
"""

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
# 全局配置
# ============================================================
DATA_DIR = '../data/'
TRANSACTION_COST_RATE = 0.0001   # 单边交易成本 0.01%（含佣金+印花税等）
SLIPPAGE_POINTS = 1              # 滑点：1个指数点
HSI_MULTIPLIER = 50              # 恒指期货每点50港元
INITIAL_CAPITAL = 5_000_000      # 初始资金500万（见风险管理部分）
N_FOLDS = 5                      # K-fold折数（TimeSeriesSplit）

# ============================================================
# 1. 数据加载与合并
# ============================================================

def load_and_merge_data(dataset_type):
    """
    加载并合并 train + val 数据。
    dataset_type: 'original' | 'day' | 'night'
    返回合并后的 DataFrame，按时间排序。
    """
    train_path = os.path.join(DATA_DIR, f'{dataset_type}_train.csv')
    val_path = os.path.join(DATA_DIR, f'{dataset_type}_validation.csv')
    test_path = os.path.join(DATA_DIR, f'{dataset_type}_test.csv')

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # 合并 train + val
    merged = pd.concat([train, val], ignore_index=True)
    
    # 标记数据来源
    merged['dataset'] = 'train_val'
    test['dataset'] = 'test'

    # 构建时间列并排序
    for df in [merged, test]:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df.sort_values('datetime', inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"[{dataset_type}] train+val: {len(merged)}, test: {len(test)}")
    return merged, test


# ============================================================
# 2. 特征工程
# ============================================================

def compute_features(df):
    """
    对原始数据进行特征工程。
    所有特征仅使用当前及历史信息，无未来信息泄露。
    
    返回带特征和标签的 DataFrame（会丢弃 NaN 行）。
    """
    df = df.copy()
    
    # --- 基础价格 ---
    o = df['hi1_open'].values.astype(float)
    h = df['hi1_high'].values.astype(float)
    l = df['hi1_low'].values.astype(float)
    c = df['hi1_close'].values.astype(float)
    v = df['hi1_volume'].values.astype(float)
    
    # --- 收益率（使用 close-to-close）---
    df['ret_1'] = np.nan
    df['ret_1'].values[1:] = (c[1:] - c[:-1]) / c[:-1]
    
    # 对数收益率
    df['log_ret_1'] = np.nan
    df['log_ret_1'].values[1:] = np.log(c[1:] / c[:-1])
    
    # 多周期滞后收益率
    for lag in [2, 3, 5, 10, 15, 30]:
        ret = np.full(len(df), np.nan)
        ret[lag:] = (c[lag:] - c[:-lag]) / c[:-lag]
        df[f'ret_{lag}'] = ret
    
    # --- 波动率（滚动标准差）---
    for window in [5, 10, 15, 30]:
        df[f'vol_{window}'] = df['ret_1'].rolling(window).std()
    
    # --- 成交量特征 ---
    df['log_volume'] = np.log(v + 1)
    df['vol_ma_5'] = df['hi1_volume'].rolling(5).mean()
    df['vol_ma_10'] = df['hi1_volume'].rolling(10).mean()
    df['vol_ratio_5'] = df['hi1_volume'] / (df['vol_ma_5'] + 1e-8)
    df['vol_ratio_10'] = df['hi1_volume'] / (df['vol_ma_10'] + 1e-8)
    
    # --- 价格形态 ---
    df['high_low_spread'] = (h - l) / (c + 1e-8)
    df['close_open_ratio'] = (c - o) / (o + 1e-8)
    df['upper_shadow'] = (h - np.maximum(o, c)) / (c + 1e-8)
    df['lower_shadow'] = (np.minimum(o, c) - l) / (c + 1e-8)
    
    # --- 移动平均 ---
    for window in [5, 10, 20, 60]:
        df[f'ma_{window}'] = df['hi1_close'].rolling(window).mean()
    
    # MA 偏离度
    for window in [5, 10, 20, 60]:
        df[f'ma_dev_{window}'] = (df['hi1_close'] - df[f'ma_{window}']) / (df[f'ma_{window}'] + 1e-8)
    
    # MA 交叉（短期/长期）
    df['ma_cross_5_20'] = df['ma_5'] / (df['ma_20'] + 1e-8) - 1
    df['ma_cross_10_60'] = df['ma_10'] / (df['ma_60'] + 1e-8) - 1
    
    # --- RSI 类指标 ---
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    for window in [6, 14, 24]:
        avg_gain = pd.Series(gain).rolling(window).mean().values
        avg_loss = pd.Series(loss).rolling(window).mean().values
        rs = avg_gain / (avg_loss + 1e-8)
        df[f'rsi_{window}'] = 100 - 100 / (1 + rs)
    
    # --- 时间特征 ---
    df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Mon
    
    # 周期性编码
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / (24 * 60))
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / (24 * 60))
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # --- 标签：下一分钟收益率（close-to-close）---
    df['target'] = np.nan
    df['target'].values[:-1] = (c[1:] - c[:-1]) / c[:-1]
    
    # 方向标签（用于辅助评估）
    df['target_direction'] = (df['target'] > 0).astype(int)
    
    # 丢弃 NaN（前几行由于滚动窗口产生）
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def get_feature_columns(df):
    """返回特征列名（排除 target、非特征列）"""
    exclude = ['date', 'time', 'timestamp', 'datetime', 'dataset',
               'target', 'target_direction',
               'hi1_open', 'hi1_high', 'hi1_low', 'hi1_close', 'hi1_volume']
    return [col for col in df.columns if col not in exclude]


# ============================================================
# 3. K-fold 交叉验证 & 超参数筛选
# ============================================================

def hyperparameter_tuning(X, y, dataset_name):
    """
    使用 TimeSeriesSplit 做 K-fold CV，筛选 LGBM 超参数。
    返回最佳参数。
    """
    # 预定义候选参数网格
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
# 4. 模型训练
# ============================================================

def train_model(X_train, y_train, params, dataset_name):
    """训练 LGBM 模型"""
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
    print(f"  [{dataset_name}] 模型训练完成")
    return model


# ============================================================
# 5. 回测与评估
# ============================================================

def simulate_trading(predictions, df_test, transaction_cost_rate=TRANSACTION_COST_RATE,
                     slippage_points=SLIPPAGE_POINTS, multiplier=HSI_MULTIPLIER):
    """
    基于预测收益率进行模拟交易。
    
    策略：
    - 预测收益率 > 成本阈值 → 做多 (position = 1)
    - 预测收益率 < -成本阈值 → 做空 (position = -1)
    - 否则平仓/不交易 (position = 0)
    
    返回：
    - trade_log: 每笔交易记录
    - equity_curve: 资金曲线
    - metrics: 指标字典
    """
    actual_returns = df_test['target'].values
    close_prices = df_test['hi1_close'].values
    
    # 成本阈值：交易成本 + 滑点成本
    # 滑点换算为收益率：slippage_points / close_price
    cost_threshold = transaction_cost_rate * 2  # 双边成本（开仓+平仓）
    
    positions = np.zeros(len(predictions))
    for i in range(len(predictions)):
        slippage_ret = slippage_points / (close_prices[i] + 1e-8)
        total_cost = cost_threshold + slippage_ret * 2  # 双边滑点
        if predictions[i] > total_cost:
            positions[i] = 1
        elif predictions[i] < -total_cost:
            positions[i] = -1
        else:
            positions[i] = 0
    
    # 计算策略收益率（含成本）
    strategy_returns = np.zeros(len(predictions))
    prev_position = 0
    trade_log = []
    trade_id = 0
    
    for i in range(len(predictions)):
        gross_ret = positions[i] * actual_returns[i]
        
        # 交易成本（仅在换仓时产生）
        turnover_cost = 0
        if positions[i] != prev_position:
            # 平旧仓 + 开新仓的成本
            if prev_position != 0:
                turnover_cost += transaction_cost_rate  # 平仓
                slippage_cost = slippage_points / (close_prices[i] + 1e-8)
                turnover_cost += slippage_cost
            if positions[i] != 0:
                turnover_cost += transaction_cost_rate  # 开仓
                slippage_cost = slippage_points / (close_prices[i] + 1e-8)
                turnover_cost += slippage_cost
            
            # 记录交易
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
    
    # 处理持仓到期末的平仓
    if prev_position != 0:
        last_cost = transaction_cost_rate
        last_cost += slippage_points / (close_prices[-1] + 1e-8)
        strategy_returns[-1] -= last_cost
    
    # 累积收益
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    
    return positions, strategy_returns, cumulative_returns, trade_log


def calculate_metrics(strategy_returns, positions):
    """
    计算评估指标：
    - Log Return (总对数收益)
    - Sharpe Ratio (年化)
    - Max Drawdown
    - 胜率 (Win Rate)
    - 盈亏比 (Profit Factor)
    - 年化收益率
    - Calmar Ratio
    """
    n = len(strategy_returns)
    
    # Log Return
    log_return = np.sum(np.log(1 + np.clip(strategy_returns, -0.99, None)))
    
    # Total Return
    total_return = np.exp(log_return) - 1
    
    # Annualized Return (假设每年约 250 个交易日 * 约 330 分钟 ≈ 82500 分钟)
    # 实际计算基于样本量：每年 ≈ 250天 * 每日交易分钟数
    # 使用广义方法：假设每年 trading minutes ≈ 250 * 6.5 * 60 ≈ 97500（含夜盘更多）
    # 简化：每年约 100,000 分钟
    annual_factor = 100_000 / n  # 年化系数
    annual_return = (1 + total_return) ** annual_factor - 1
    
    # Sharpe Ratio (年化)
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    if std_ret > 0:
        sharpe_ratio = mean_ret / std_ret * np.sqrt(n) * np.sqrt(annual_factor / (100_000 / n))
        # 简化：sharpe_ratio = mean_ret / std_ret * sqrt(100000)
        sharpe_ratio = mean_ret / std_ret * np.sqrt(100_000)
    else:
        sharpe_ratio = 0
    
    # Max Drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Win Rate
    trades = positions != 0
    if np.sum(trades) > 0:
        win_rate = np.mean(strategy_returns[trades] > 0)
    else:
        win_rate = 0
    
    # Profit Factor
    positive_rets = strategy_returns[trades][strategy_returns[trades] > 0]
    negative_rets = strategy_returns[trades][strategy_returns[trades] < 0]
    if len(negative_rets) > 0 and np.sum(np.abs(negative_rets)) > 0:
        profit_factor = np.sum(positive_rets) / np.sum(np.abs(negative_rets))
    else:
        profit_factor = np.inf if len(positive_rets) > 0 else 0
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Average trade return
    if np.sum(trades) > 0:
        avg_trade_return = np.mean(strategy_returns[trades])
    else:
        avg_trade_return = 0
    
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
    }
    
    return metrics


def trade_log_count(positions):
    """计算交易次数（仓位变动次数）"""
    changes = np.diff(positions, prepend=0)
    return np.sum(changes != 0)


def evaluate_on_test(model, X_test, df_test, scenario_name):
    """在测试集上评估模型，返回指标和交易日志"""
    predictions = model.predict(X_test)
    positions, strategy_rets, cum_rets, trade_log = simulate_trading(predictions, df_test)
    metrics = calculate_metrics(strategy_rets, positions)
    
    print(f"\n  [{scenario_name}]")
    print(f"    Log Return:       {metrics['Log Return']:.4f}")
    print(f"    Total Return:     {metrics['Total Return']:.4%}")
    print(f"    Annualized Return:{metrics['Annualized Return']:.4%}")
    print(f"    Sharpe Ratio:     {metrics['Sharpe Ratio']:.4f}")
    print(f"    Max Drawdown:     {metrics['Max Drawdown']:.4%}")
    print(f"    Win Rate:         {metrics['Win Rate']:.4%}")
    print(f"    Profit Factor:    {metrics['Profit Factor']:.4f}")
    print(f"    Calmar Ratio:     {metrics['Calmar Ratio']:.4f}")
    print(f"    Total Trades:     {metrics['Total Trades']}")
    
    return predictions, positions, strategy_rets, cum_rets, trade_log, metrics


# ============================================================
# 6. 主流程
# ============================================================

def main():
    print("=" * 80)
    print("  恒指期货 LGBM 模型训练与评估")
    print("=" * 80)
    
    # --- 存储所有结果 ---
    all_models = {}
    all_params = {}
    all_features = {}      # feature columns per dataset
    all_test_data = {}
    all_test_features = {}
    
    # --- 每个数据集：加载、特征工程、CV、训练 ---
    for ds in ['original', 'day', 'night']:
        print(f"\n{'='*60}")
        print(f"  处理 {ds.upper()} 数据集")
        print(f"{'='*60}")
        
        # 1. 加载合并
        merged, test = load_and_merge_data(ds)
        
        # 2. 特征工程
        print(f"  特征工程中...")
        merged_feat = compute_features(merged)
        test_feat = compute_features(test)
        
        feature_cols = get_feature_columns(merged_feat)
        all_features[ds] = feature_cols
        
        X_train_val = merged_feat[feature_cols]
        y_train_val = merged_feat['target']
        X_test = test_feat[feature_cols]
        y_test = test_feat['target']
        
        all_test_data[ds] = test_feat
        all_test_features[ds] = X_test
        
        print(f"  特征数: {len(feature_cols)}, 训练集: {len(X_train_val)}, 测试集: {len(X_test)}")
        
        # 3. K-fold CV 超参数筛选
        best_params, cv_score = hyperparameter_tuning(X_train_val, y_train_val, ds)
        all_params[ds] = best_params
        
        # 4. 训练最终模型
        print(f"\n  训练最终模型...")
        model = train_model(X_train_val, y_train_val, best_params, ds)
        all_models[ds] = model
    
    # --- 测试集评估：5种场景 ---
    print(f"\n{'='*80}")
    print(f"  测试集评估")
    print(f"{'='*80}")
    
    results = {}
    
    scenarios = [
        ('original_model on original_test', all_models['original'],
         all_test_features['original'], all_test_data['original']),
        ('original_model on day_test', all_models['original'],
         all_test_features['day'], all_test_data['day']),
        ('original_model on night_test', all_models['original'],
         all_test_features['night'], all_test_data['night']),
        ('day_model on day_test', all_models['day'],
         all_test_features['day'], all_test_data['day']),
        ('night_model on night_test', all_models['night'],
         all_test_features['night'], all_test_data['night']),
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
    
    # --- 比较分析 ---
    print(f"\n{'='*80}")
    print(f"  模型比较分析")
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
        }
        for name, res in results.items()
    }).T
    
    print(comparison.to_string())
    comparison.to_csv('model_comparison.csv')
    print("\n  比较结果已保存至 model_comparison.csv")
    
    # --- 选择最佳模型 ---
    # 综合考虑：Sharpe Ratio, Max Drawdown, Profit Factor
    scores = {}
    for name, res in results.items():
        m = res['metrics']
        # 综合评分（加权）：Sharpe > Calmar > Win Rate
        score = (m['Sharpe Ratio'] * 0.4 + 
                 m['Calmar Ratio'] * 0.3 + 
                 m['Profit Factor'] * 0.2 + 
                 m['Win Rate'] * 0.1)
        scores[name] = score
    
    best_model_name = max(scores, key=scores.get)
    print(f"\n  ===> 最佳模型: {best_model_name}")
    print(f"  综合评分: {scores[best_model_name]:.4f}")
    
    # --- 保存最佳模型信息 ---
    best_info = {
        'best_model': best_model_name,
        'scores': {k: float(v) for k, v in scores.items()},
        'best_metrics': {k: float(v) if not isinstance(v, str) else v
                         for k, v in results[best_model_name]['metrics'].items()},
    }
    
    with open('best_model_info.json', 'w') as f:
        json.dump(best_info, f, indent=2, default=str)
    
    print(f"\n  最佳模型信息已保存至 best_model_info.json")
    
    # --- 保存所有模型参数 ---
    all_info = {
        'transaction_cost_rate': TRANSACTION_COST_RATE,
        'slippage_points': SLIPPAGE_POINTS,
        'hsi_multiplier': HSI_MULTIPLIER,
        'n_folds': N_FOLDS,
        'params': {k: {pk: str(pv) if not isinstance(pv, (int, float, bool)) else pv
                       for pk, pv in v.items()}
                   for k, v in all_params.items()},
        'feature_counts': {k: len(v) for k, v in all_features.items()},
        'feature_lists': all_features,
    }
    
    with open('all_model_params.json', 'w') as f:
        json.dump(all_info, f, indent=2, default=str)
    
    print(f"\n  所有模型参数已保存至 all_model_params.json")
    
    # --- 保存最佳模型 ---
    if 'original' in best_model_name:
        best_model = all_models['original']
    elif 'day' in best_model_name:
        best_model = all_models['day']
    else:
        best_model = all_models['night']
    
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    print(f"  最佳模型已保存至 best_model.pkl")
    
    return all_models, results, best_model_name


if __name__ == '__main__':
    all_models, results, best_model_name = main()
