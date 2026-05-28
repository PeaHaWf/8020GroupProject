

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')


INITIAL_CAPITAL = 5_000_000
TRANSACTION_COST_RATE = 0.0001
SLIPPAGE_POINTS = 1
HSI_MULTIPLIER = 50
CONTRACT_MARGIN_RATE = 0.05


MAX_POSITION_RATIO = 0.30
MAX_SINGLE_LOSS_RATIO = 0.02
MAX_DAILY_DRAWDOWN_RATIO = 0.05
MAX_CONSECUTIVE_LOSSES = 5
VOL_TARGET_RATIO = 0.01

DATA_DIR = '../data/'


# ============================================================
# risk management
# ============================================================

class RiskManager:

    
    def __init__(self, initial_capital=INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        

        self.max_position_ratio = MAX_POSITION_RATIO
        self.max_single_loss_ratio = MAX_SINGLE_LOSS_RATIO
        self.max_daily_drawdown_ratio = MAX_DAILY_DRAWDOWN_RATIO
        self.max_consecutive_losses = MAX_CONSECUTIVE_LOSSES
        self.vol_target_ratio = VOL_TARGET_RATIO
        

        self.current_position = 0
        self.current_contracts = 0
        self.entry_price = 0
        self.entry_capital = 0
        

        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.is_halted = False
        

        self.trade_log = []
        

        self.equity_curve = []
        self.drawdown_curve = []
        

        self.current_date = None
    
    def reset_daily(self, new_date):

        if self.current_date != new_date:
            self.current_date = new_date
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0
            self.is_halted = False
            self.consecutive_losses = 0
    
    def calculate_position_size(self, price, volatility, predicted_return):

        if self.is_halted:
            return 0
        

        contract_value = price * HSI_MULTIPLIER
        

        max_contracts_by_capital = int(
            self.current_capital * self.max_position_ratio / contract_value
        )
        

        if volatility > 0:
            # Kelly-like sizing
            vol_adjusted_ratio = self.vol_target_ratio / (volatility + 1e-8)
            vol_adjusted_ratio = min(vol_adjusted_ratio, 1.0)  # 上限100%
            contracts_by_vol = int(
                self.current_capital * vol_adjusted_ratio * self.max_position_ratio / contract_value
            )
        else:
            contracts_by_vol = 0
        

        contracts = min(max_contracts_by_capital, max(contracts_by_vol, 1))
        contracts = max(1, contracts) if contracts > 0 else 0
        
        return contracts
    
    def check_risk_limits(self, potential_loss_points):

        potential_loss_amount = abs(potential_loss_points) * HSI_MULTIPLIER * max(self.current_contracts, 1)
        max_allowed_loss = self.current_capital * self.max_single_loss_ratio
        return potential_loss_amount <= max_allowed_loss
    
    def check_daily_drawdown(self):

        if self.daily_start_capital > 0:
            daily_dd = (self.current_capital - self.daily_start_capital) / self.daily_start_capital
            if daily_dd < -self.max_daily_drawdown_ratio:
                return False  # 触发日内回撤限制
        return True
    
    def check_consecutive_losses(self):

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.is_halted = True
            return False
        return not self.is_halted
    
    def open_position(self, timestamp, price, direction, volatility, predicted_return):

        if self.is_halted:
            return False
        

        if not self.check_consecutive_losses():
            return False
        

        if not self.check_daily_drawdown():
            self.is_halted = True
            return False
        

        contracts = self.calculate_position_size(price, volatility, predicted_return)
        if contracts == 0:
            return False
        

        if self.current_position != 0 and self.current_position != direction:
            self.close_position(timestamp, price, reason='reverse')
        

        self.current_position = direction
        self.current_contracts = contracts
        self.entry_price = price
        self.entry_capital = self.current_capital
        

        cost_rate = TRANSACTION_COST_RATE + SLIPPAGE_POINTS / (price + 1e-8)
        cost_amount = cost_rate * contracts * price * HSI_MULTIPLIER
        self.current_capital -= cost_amount
        
        self.total_trades += 1
        

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

        if self.current_position == 0:
            return 0
        

        if self.current_position == 1:
            pnl_points = price - self.entry_price
        else:
            pnl_points = self.entry_price - price

        cost_rate = TRANSACTION_COST_RATE + SLIPPAGE_POINTS / (price + 1e-8)
        cost_amount = cost_rate * self.current_contracts * price * HSI_MULTIPLIER
        

        gross_pnl = pnl_points * HSI_MULTIPLIER * self.current_contracts
        net_pnl = gross_pnl - cost_amount
        

        self.current_capital += net_pnl
        

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        

        self.daily_pnl += pnl_points * self.current_contracts
        

        if net_pnl < 0:
            self.consecutive_losses += 1
            self.losing_trades += 1
        else:
            self.consecutive_losses = 0
            self.winning_trades += 1
        

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
        

        self.current_position = 0
        self.current_contracts = 0
        self.entry_price = 0
        
        return net_pnl
    
    def record_equity(self, timestamp):

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


        trade_log_df = pd.DataFrame(self.trade_log)
        trade_log_df.to_csv(os.path.join(output_dir, 'trade_log.csv'), index=False)
        print(f"  log saved to trade_log.csv ({len(trade_log_df)} records)")
        

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)
        print(f"  equity saved to equity_curve.csv ({len(equity_df)} records)")
        

        summary = self.finalize()
        with open(os.path.join(output_dir, 'risk_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  risk saved to risk_summary.json")
        
        return summary


# ============================================================
# simulation trade
# ============================================================

def run_backtest_with_risk_management():

    print("=" * 80)
    print(" simulating trading with risk management")
    print("=" * 80)
    print(f"  initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  max position ratio: {MAX_POSITION_RATIO*100:.0f}%")
    print(f"  max single loss ratio: {MAX_SINGLE_LOSS_RATIO*100:.1f}%")
    print(f"  max daily drawdown ratio: {MAX_DAILY_DRAWDOWN_RATIO*100:.0f}%")
    print(f"  max consecutive losses: {MAX_CONSECUTIVE_LOSSES}")
    print()
    
    
    model_path = 'best_model.pkl'
    if not os.path.exists(model_path):
       
        print("  best_model.pklnot found, please run model_training.py first")
        return
    
    model = joblib.load(model_path)
    print(f"  模型已加载: {model_path}")
    
    # load test data and compute features
    from model_training import load_and_merge_data, compute_features, get_feature_columns
    
    _, test_data = load_and_merge_data('night')
    test_feat = compute_features(test_data)
    feature_cols = get_feature_columns(test_feat)
    X_test = test_feat[feature_cols]
    
    print(f"  test data: {len(test_feat)} records")
    

    predictions = model.predict(X_test)
    test_feat['predicted_return'] = predictions
    

    test_feat['volatility'] = test_feat['vol_5'].fillna(0)
    
    
    rm = RiskManager(initial_capital=INITIAL_CAPITAL)
    
    record_interval = 1
    
    base_cost_threshold = TRANSACTION_COST_RATE * 2  
    
    print(f"\n  starting simulation...")
    
    prev_position = 0
    
    for i in range(len(test_feat)):
        row = test_feat.iloc[i]
        timestamp = row['datetime']
        close_price = row['hi1_close']
        pred_ret = row['predicted_return']
        volatility = row['volatility']
        current_date = str(timestamp.date())
        
        rm.reset_daily(current_date)
        
        slippage_ret = SLIPPAGE_POINTS / (close_price + 1e-8)
        total_cost_threshold = base_cost_threshold + slippage_ret * 2
        
        if pred_ret > total_cost_threshold:
            desired_position = 1  # long
        elif pred_ret < -total_cost_threshold:
            desired_position = -1  # short
        else:
            desired_position = 0  # no trade
        
        # conduct risk checks and execute trades
        if desired_position != rm.current_position:
            # first close existing position if any
            if rm.current_position != 0:
                rm.close_position(timestamp, close_price, 
                                  reason='reverse' if desired_position != 0 else 'signal')
            
            # then open new position if desired
            if desired_position != 0:
                rm.open_position(timestamp, close_price, desired_position, 
                                 volatility, pred_ret)
        
        # record equity at specified intervals
        if i % record_interval == 0:
            rm.record_equity(timestamp)
    
    # force close any open position at the end of the test period
    if rm.current_position != 0:
        last_row = test_feat.iloc[-1]
        rm.close_position(last_row['datetime'], last_row['hi1_close'], reason='end_of_period')
        rm.record_equity(last_row['datetime'])
    
    summary = rm.finalize()
    
    print(f"\n{'='*60}")
    print(f"  simulation completed")
    print(f"{'='*60}")
    print(f"  initial capital:     ${summary['initial_capital']:,.0f}")
    print(f"  final capital:     ${summary['final_capital']:,.0f}")
    print(f"  total PnL:       ${summary['total_pnl']:,.0f}")
    print(f"  total return:     {summary['total_return']:.4%}")
    print(f"  peak capital:     ${summary['peak_capital']:,.0f}")
    print(f"  max drawdown:     {summary['max_drawdown']:.4%}")
    print(f"  total trades:   {summary['total_trades']}")
    print(f"  winning trades:     {summary['winning_trades']}")
    print(f"  losing trades:     {summary['losing_trades']}")
    print(f"  win rate:         {summary['win_rate']:.4%}")
    print(f"  was halted: {summary['was_halted']}")
    
    # save results to files
    rm.save_results()
    
    trades_df = pd.DataFrame(rm.trade_log)
    close_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()
    
    if len(close_trades) > 0:
        print(f"\n{'='*60}")
        print(f"  per-trade PnL details")
        print(f"{'='*60}")
        
        # group by trade pairs (each OPEN+CLOSE)
        print(f"\n  --- first 10 complete trades ---")
        display_count = min(10, len(close_trades))
        for i in range(display_count):
            t = close_trades.iloc[i]
            print(f"  交易#{i+1}: {t['direction']} | PnL=${t['net_pnl']:,.0f} "
                  f"({t['pnl_points']:+.0f}points) | capital remaining=${t['capital_after']:,.0f}")
        
        if len(close_trades) > 10:
            print(f"\n  --- last 10 trades ---")
            for i in range(max(0, len(close_trades)-10), len(close_trades)):
                t = close_trades.iloc[i]
                print(f"  交易#{i+1}: {t['direction']} | PnL=${t['net_pnl']:,.0f} "
                      f"({t['pnl_points']:+.0f}points) | capital remaining=${t['capital_after']:,.0f}")
    
    return rm, summary


if __name__ == '__main__':
    rm, summary = run_backtest_with_risk_management()
