"""
风险管理系统
========================
功能：
1. 初始资金500万，实现风险管理系统
2. 使用实测模型 (original_model_LGBM) 在测试集上做决策
3. 计算每次交易的盈亏情况
4. 计算每次交易后的资金余额

风险控制模块包括：
- 单笔最大亏损限制
- 最大持仓量限制
- 日内最大回撤限制
- 连续亏损熔断
- 头寸规模计算（基于波动率调整）
"""

import sys
import io
# 强制UTF-8输出（解决Windows GBK编码问题）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数（可在此修改）
# ============================================================
INITIAL_CAPITAL = 5_000_000        # 初始资金500万元
TRANSACTION_COST_RATE = 0.0001     # 单边交易成本 0.01%
SLIPPAGE_POINTS = 1                # 滑点：1个指数点
HSI_MULTIPLIER = 50                # 恒指期货每点50港元
CONTRACT_MARGIN_RATE = 0.05        # 保证金比例 5%

# 风控参数
MAX_POSITION_RATIO = 0.30          # 最大持仓占资金比例 30%
MAX_SINGLE_LOSS_RATIO = 0.02       # 单笔最大亏损占资金比例 2%
MAX_DAILY_DRAWDOWN_RATIO = 0.05    # 修改位置日内最大回撤限制 5%
MAX_CONSECUTIVE_LOSSES = 5         # 连续亏损熔断次数
VOL_TARGET_RATIO = 0.01            # 目标波动率（每笔交易）1%

DATA_DIR = './data/'


# ============================================================
# 风险管理系统
# ============================================================

class RiskManager:
    """
    风险管理系统
    
    职责：
    1. 管理资金和头寸
    2. 执行风控规则
    3. 记录每笔交易
    4. 计算实时资金余额
    """
    
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        
        # 风控参数
        self.max_position_ratio = MAX_POSITION_RATIO
        self.max_single_loss_ratio = MAX_SINGLE_LOSS_RATIO
        self.max_daily_drawdown_ratio = MAX_DAILY_DRAWDOWN_RATIO
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        self.vol_target_ratio = VOL_TARGET_RATIO
        
        # 状态跟踪
        self.current_position = 0       # 当前持仓方向：1多头/-1空头/0空仓
        self.current_contracts = 0      # 当前持仓手数
        self.entry_price = 0            # 入场价格
        self.entry_capital = 0          # 入场时资金
        
        # 统计
        self.consecutive_losses = 0     # 连续亏损次数
        self.daily_pnl = 0              # 日内累计盈亏（点数）
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.is_halted = False          # 是否熔断
        
        # 交易日志
        self.trade_log = []
        
        # 资金曲线
        self.equity_curve = []
        self.drawdown_curve = []
        
        # 当前日期（用于日内风控）
        self.current_date = None
    
    def reset_daily(self, new_date):
        """重置日内统计"""
        if self.current_date != new_date:
            self.current_date = new_date
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0
            self.is_halted = False  # 每日重置熔断状态
            self.consecutive_losses = 0  # 每日重置连续亏损计数
    
    def calculate_position_size(self, price, volatility, predicted_return):
        """
        计算头寸规模
        
        基于：
        1. 资金管理：最大持仓比例限制
        2. 波动率调整：目标波动率
        3. 信号强度：预测收益率的置信度
        
        返回：合约手数（整数）
        """
        if self.is_halted:
            return 0
        
        # 每手合约价值（港元）
        contract_value = price * HSI_MULTIPLIER
        
        # 基于最大持仓比例的手数上限
        max_contracts_by_capital = int(
            self.current_capital * self.max_position_ratio / contract_value
        )
        
        # 基于波动率调整的手数
        if volatility > 0:
            # Kelly-like sizing: 目标波动率 / 实际波动率
            vol_adjusted_ratio = self.vol_target_ratio / (volatility + 1e-8)
            vol_adjusted_ratio = min(vol_adjusted_ratio, 1.0)  # 上限100%
            contracts_by_vol = int(
                self.current_capital * vol_adjusted_ratio * self.max_position_ratio / contract_value
            )
        else:
            contracts_by_vol = 0
        
        # 取两者中较小的
        contracts = min(max_contracts_by_capital, max(contracts_by_vol, 1))
        contracts = max(1, contracts) if contracts > 0 else 0
        
        return contracts
    
    def check_risk_limits(self, potential_loss_points):
        """检查单笔最大亏损限制"""
        potential_loss_amount = abs(potential_loss_points) * HSI_MULTIPLIER * max(self.current_contracts, 1)
        max_allowed_loss = self.current_capital * self.max_single_loss_ratio
        return potential_loss_amount <= max_allowed_loss
    
    def check_daily_drawdown(self):
        """检查日内最大回撤"""
        if self.daily_start_capital > 0:
            daily_dd = (self.current_capital - self.daily_start_capital) / self.daily_start_capital
            if daily_dd < -self.max_daily_drawdown_ratio:
                return False  # 触发日内回撤限制
        return True
    
    def check_consecutive_losses(self):
        """检查连续亏损熔断"""
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_halted = True
            return False
        return not self.is_halted
    
    def open_position(self, timestamp, price, direction, volatility, predicted_return):
        """
        开仓
        
        参数：
        - timestamp: 时间戳
        - price: 当前价格
        - direction: 1=做多, -1=做空
        - volatility: 当前波动率
        - predicted_return: 模型预测收益率
        
        返回：是否成功开仓
        """
        # 检查熔断
        if self.is_halted:
            return False
        
        # 检查连续亏损
        if not self.check_consecutive_losses():
            return False
        
        # 检查日内回撤
        if not self.check_daily_drawdown():
            self.is_halted = True
            return False
        
        # 计算手数
        contracts = self.calculate_position_size(price, volatility, predicted_return)
        if contracts == 0:
            return False
        
        # 平掉现有反向仓位
        if self.current_position != 0 and self.current_position != direction:
            self.close_position(timestamp, price, reason='reverse')
        
        # 开仓
        self.current_position = direction
        self.current_contracts = contracts
        self.entry_price = price
        self.entry_capital = self.current_capital
        
        # 扣除开仓交易成本
        cost_rate = TRANSACTION_COST_RATE + SLIPPAGE_POINTS / (price + 1e-8)
        cost_amount = cost_rate * contracts * price * HSI_MULTIPLIER
        self.current_capital -= cost_amount
        
        self.total_trades += 1
        
        # 记录交易
        self.trade_log.append({
            'timestamp': str(timestamp),
            'action': 'OPEN',
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'price': price,
            'contracts': contracts,
            'cost': cost_amount,
            'capital_after': self.current_capital,
            'predicted_return': predicted_return,
            'volatility': volatility,
        })
        
        return True
    
    def close_position(self, timestamp, price, reason='signal'):
        """
        平仓
        
        返回：盈亏点数
        """
        if self.current_position == 0:
            return 0
        
        # 计算盈亏（点数）
        if self.current_position == 1:  # 多头
            pnl_points = price - self.entry_price
        else:  # 空头
            pnl_points = self.entry_price - price
        
        # 扣除平仓交易成本
        cost_rate = TRANSACTION_COST_RATE + SLIPPAGE_POINTS / (price + 1e-8)
        cost_amount = cost_rate * self.current_contracts * price * HSI_MULTIPLIER
        
        # 计算净盈亏（金额）
        gross_pnl = pnl_points * HSI_MULTIPLIER * self.current_contracts
        net_pnl = gross_pnl - cost_amount
        
        # 更新资金
        self.current_capital += net_pnl
        
        # 更新峰值资金
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # 更新日内累计
        self.daily_pnl += pnl_points * self.current_contracts
        
        # 更新连续亏损
        if net_pnl < 0:
            self.consecutive_losses += 1
            self.losing_trades += 1
        else:
            self.consecutive_losses = 0
            self.winning_trades += 1
        
        # 记录交易
        self.trade_log.append({
            'timestamp': str(timestamp),
            'action': 'CLOSE',
            'direction': 'LONG' if self.current_position == 1 else 'SHORT',
            'price': price,
            'contracts': self.current_contracts,
            'pnl_points': pnl_points,
            'gross_pnl': gross_pnl,
            'cost': cost_amount,
            'net_pnl': net_pnl,
            'capital_after': self.current_capital,
            'reason': reason,
            'consecutive_losses': self.consecutive_losses,
        })
        
        # 重置仓位
        self.current_position = 0
        self.current_contracts = 0
        self.entry_price = 0
        
        return net_pnl
    
    def record_equity(self, timestamp):
        """记录当前时间点的资金情况"""
        unrealized_pnl = 0
        drawdown = (self.current_capital - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0
        
        self.equity_curve.append({
            'timestamp': str(timestamp),
            'capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': drawdown,
            'position': self.current_position,
            'contracts': self.current_contracts,
            'is_halted': self.is_halted,
        })
    
    def finalize(self):
        """最终清算"""
        summary = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'total_pnl': self.current_capital - self.initial_capital,
            'peak_capital': self.peak_capital,
            'max_drawdown': min([e['drawdown'] for e in self.equity_curve]) if self.equity_curve else 0,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'was_halted': self.is_halted,
        }
        return summary
    
    def save_results(self, output_dir='./'):
        """保存交易日志和资金曲线"""
        # 交易日志
        trade_log_df = pd.DataFrame(self.trade_log)
        trade_log_df.to_csv(os.path.join(output_dir, 'trade_log.csv'), index=False)
        print(f"  交易日志保存至 trade_log.csv ({len(trade_log_df)} 条记录)")
        
        # 资金曲线
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)
        print(f"  资金曲线保存至 equity_curve.csv ({len(equity_df)} 条记录)")
        
        # 汇总
        summary = self.finalize()
        with open(os.path.join(output_dir, 'risk_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  风控汇总保存至 risk_summary.json")
        
        return summary


# ============================================================
# 模拟交易主流程
# ============================================================

def run_backtest_with_risk_management():
    """
    使用best_model (original_model_LGBM) 在 original_test 上进行模拟交易
    """
    print("=" * 80)
    print("  风险管理系统 - 模拟交易")
    print("=" * 80)
    print(f"  初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print(f"  最大持仓比例: {MAX_POSITION_RATIO*100:.0f}%")
    print(f"  单笔最大亏损: {MAX_SINGLE_LOSS_RATIO*100:.1f}%")
    print(f"  日内最大回撤限制: {MAX_DAILY_DRAWDOWN_RATIO*100:.0f}%")
    print(f"  连续亏损熔断: {MAX_CONSECUTIVE_LOSSES}次")
    print()
    
    # 加载模型
    model_path = 'best_model.pkl'
    if not os.path.exists(model_path):
        # 如果没有best_model.pkl，尝试加载original trained模型
        print("  best_model.pkl未找到，请先运行model_training.py")
        return
    
    model = joblib.load(model_path)
    print(f"  模型已加载: {model_path}")
    
    # 加载original_test数据
    from model_training import load_and_merge_data, compute_features, get_feature_columns
    
    _, test_data = load_and_merge_data('original')
    test_feat = compute_features(test_data)
    feature_cols = get_feature_columns(test_feat)
    X_test = test_feat[feature_cols]
    
    print(f"  测试数据: {len(test_feat)} 条记录")
    
    # 模型预测
    predictions = model.predict(X_test)
    test_feat['predicted_return'] = predictions
    
    # 使用5分钟滚动窗口波动率
    test_feat['volatility'] = test_feat['vol_5'].fillna(0)
    
    # 初始化风控管理器
    rm = RiskManager(initial_capital=INITIAL_CAPITAL)
    
    # 每N条记录记录一次资金曲线（避免记录过多）
    record_interval = 1
    
    # 交易决策阈值
    base_cost_threshold = TRANSACTION_COST_RATE * 2  # 双边交易成本
    
    print(f"\n  开始模拟交易...")
    
    prev_position = 0
    
    for i in range(len(test_feat)):
        row = test_feat.iloc[i]
        timestamp = row['datetime']
        close_price = row['hi1_close']
        pred_ret = row['predicted_return']
        volatility = row['volatility']
        current_date = str(timestamp.date())
        
        # 每日重置
        rm.reset_daily(current_date)
        
        # 成本阈值（含滑点）
        slippage_ret = SLIPPAGE_POINTS / (close_price + 1e-8)
        total_cost_threshold = base_cost_threshold + slippage_ret * 2
        
        # 交易决策
        if pred_ret > total_cost_threshold:
            desired_position = 1  # 做多
        elif pred_ret < -total_cost_threshold:
            desired_position = -1  # 做空
        else:
            desired_position = 0  # 不交易
        
        # 执行交易
        if desired_position != rm.current_position:
            # 先平仓
            if rm.current_position != 0:
                rm.close_position(timestamp, close_price, 
                                  reason='reverse' if desired_position != 0 else 'signal')
            
            # 再开仓
            if desired_position != 0:
                rm.open_position(timestamp, close_price, desired_position, 
                                 volatility, pred_ret)
        
        # 记录资金曲线
        if i % record_interval == 0:
            rm.record_equity(timestamp)
    
    # 期末强制平仓
    if rm.current_position != 0:
        last_row = test_feat.iloc[-1]
        rm.close_position(last_row['datetime'], last_row['hi1_close'], reason='end_of_period')
        rm.record_equity(last_row['datetime'])
    
    # 最终清算
    summary = rm.finalize()
    
    print(f"\n{'='*60}")
    print(f"  模拟交易完成")
    print(f"{'='*60}")
    print(f"  初始资金:     ${summary['initial_capital']:,.0f}")
    print(f"  最终资金:     ${summary['final_capital']:,.0f}")
    print(f"  总盈亏:       ${summary['total_pnl']:,.0f}")
    print(f"  总收益率:     {summary['total_return']:.4%}")
    print(f"  峰值资金:     ${summary['peak_capital']:,.0f}")
    print(f"  最大回撤:     {summary['max_drawdown']:.4%}")
    print(f"  总交易次数:   {summary['total_trades']}")
    print(f"  盈利次数:     {summary['winning_trades']}")
    print(f"  亏损次数:     {summary['losing_trades']}")
    print(f"  胜率:         {summary['win_rate']:.4%}")
    print(f"  是否触发熔断: {summary['was_halted']}")
    
    # 保存结果
    rm.save_results()
    
    # 打印交易盈亏明细（前20笔和后10笔）
    trades_df = pd.DataFrame(rm.trade_log)
    close_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()
    
    if len(close_trades) > 0:
        print(f"\n{'='*60}")
        print(f"  每笔交易盈亏明细")
        print(f"{'='*60}")
        
        # 按交易对分组（每对OPEN+CLOSE）
        print(f"\n  --- 前10笔完整交易 ---")
        display_count = min(10, len(close_trades))
        for i in range(display_count):
            t = close_trades.iloc[i]
            print(f"  交易#{i+1}: {t['direction']} | PnL=¥{t['net_pnl']:,.0f} "
                  f"({t['pnl_points']:+.0f}点) | 资金余额=¥{t['capital_after']:,.0f}")
        
        if len(close_trades) > 10:
            print(f"\n  --- 最后10笔交易 ---")
            for i in range(max(0, len(close_trades)-10), len(close_trades)):
                t = close_trades.iloc[i]
                print(f"  交易#{i+1}: {t['direction']} | PnL=¥{t['net_pnl']:,.0f} "
                      f"({t['pnl_points']:+.0f}点) | 资金余额=¥{t['capital_after']:,.0f}")
    
    return rm, summary


if __name__ == '__main__':
    rm, summary = run_backtest_with_risk_management()
