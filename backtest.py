import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CLEANED_FILE = './data/hi1_cleaned.csv'
INITIAL_CAPITAL = 200_000
MULTIPLIER = 50
COMMISSION_PER_TRADE = 50


class BaseStrategy:
    def __init__(self, name="UnnamedStrategy"):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Not implemented yet")


class BuyAndHoldStrategy(BaseStrategy):
    """Baseline策略：从第一根K线开始一直持有多仓，直到结束"""

    def __init__(self):
        super().__init__(name="BuyAndHold")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['position'] = 1.0  # 始终持有多仓
        return signals


class AdvancedBacktester:
    def __init__(self, data: pd.DataFrame, initial_capital=200000,
                 multiplier=50, commission_per_trade=50, slippage_points=1,
                 force_intraday=True):
        """
        slippage_points: 每笔交易预计滑点多少个点位（恒指1点=50HKD）
        force_intraday: 是否强制日内平仓（每天最后一分钟仓位清空），默认为True。
                        用于日内策略时应设为True，买入并持有等隔夜策略应设为False。
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.multiplier = multiplier
        self.commission_per_trade = commission_per_trade
        self.slippage_points = slippage_points
        self.force_intraday = force_intraday
        self.results = None
        self.strategy_name = None  # 记录策略名，报告时使用

    def run(self, strategy: BaseStrategy):
        self.strategy_name = strategy.name
        print(f"开始回测策略: {strategy.name}")

        # 1. 生成原始信号
        signals = strategy.generate_signals(self.data)
        if 'position' not in signals.columns:
            signals['position'] = 0.0

        self.data['raw_position'] = signals['position'].fillna(0)

        # 2. 【新增开关】仅在启用日内平仓时，才将每天最后一分钟仓位强平为0
        if self.force_intraday:
            self.data['date'] = self.data.index.date
            last_min_mask = self.data.groupby('date').cumcount(ascending=False) == 0
            self.data['position'] = np.where(last_min_mask, 0.0, self.data['raw_position'])
            # 如果执行了强平，打印提示
            print("已启用日内强制平仓（每日尾盘仓位清空）")
        else:
            # 不强制平仓，完全保留策略的原始仓位信号（允许隔夜持仓）
            self.data['position'] = self.data['raw_position']
            print("已关闭日内强制平仓，允许隔夜持仓")

        # 3. 持仓信号延后一期（严格防止未来数据）
        self.data['position'] = self.data['position'].shift(1).fillna(0)

        # 4. 计算价格变动与未扣费盈亏
        self.data['price_change'] = self.data['hi1_close'].diff().fillna(0)
        self.data['raw_pnl'] = self.data['position'] * self.data['price_change'] * self.multiplier

        # 5. 计算精确的交易手数与摩擦成本（手续费 + 滑点）
        self.data['trade_lots'] = self.data['position'].diff().abs().fillna(0)

        # 总摩擦 = (单笔手续费) + (滑点点数 * 点值)
        total_friction_per_lot = self.commission_per_trade + (self.slippage_points * self.multiplier)
        self.data['friction_cost'] = self.data['trade_lots'] * total_friction_per_lot

        # 净盈亏
        self.data['pnl'] = self.data['raw_pnl'] - self.data['friction_cost']
        self.data['cum_pnl'] = self.data['pnl'].cumsum()
        self.data['equity'] = self.initial_capital + self.data['cum_pnl']

        self.results = self.data
        self._calculate_metrics()
        return self.results

    def _calculate_metrics(self):
        equity = self.results['equity']

        # 计算实际交易日数（用于年化，更准确）
        if 'date' not in self.results.columns:
            self.results['date'] = self.results.index.date
        trading_days = self.results['date'].nunique()
        years = trading_days / 252 if trading_days > 0 else 1

        final_equity = equity.iloc[-1]
        total_return = (final_equity / self.initial_capital - 1) * 100

        if final_equity <= 0:
            ann_return = -1.0
            print("警告：最终权益为负，亏损已超过100%，年化收益率设为 -100%")
        else:
            ann_return = (final_equity / self.initial_capital) ** (1 / years) - 1

        # 基于日收益序列的波动率和夏普
        daily_pnl = self.results.groupby('date')['pnl'].sum()
        daily_returns = daily_pnl / self.initial_capital
        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol != 0 and not np.isnan(ann_return) else 0.0
        max_dd = ((equity.cummax() - equity) / equity.cummax()).max() * 100

        print("\n" + "=" * 50)
        print(f"策略回测报告: {self.strategy_name}")
        print("=" * 50)
        print(f"测试时间范围          : {self.results.index.min()} 至 {self.results.index.max()}")
        print(f"交易日数量            : {trading_days}")
        print(f"日内强制平仓          : {'是' if self.force_intraday else '否'}")
        print(f"初始资金              : HKD {self.initial_capital:,.0f}")
        print(f"最终权益              : HKD {equity.iloc[-1]:,.0f}")
        print(f"总收益率              : {total_return:.2f}%")
        print(f"年化收益率            : {ann_return * 100:.2f}%")
        print(f"年化波动率(基于日)     : {ann_vol * 100:.2f}%")
        print(f"夏普比率              : {sharpe:.2f}")
        print(f"最大回撤              : {max_dd:.2f}%")
        print(f"总交易手数（双边）    : {self.results['trade_lots'].sum():.0f} 手")
        print(f"总摩擦成本(含滑点)     : HKD {self.results['friction_cost'].sum():,.0f}")
        print("=" * 50)

    def plot(self):
        """绘制权益曲线和价格"""
        fig, ax1 = plt.subplots(figsize=(14, 8))

        # 价格
        ax1.plot(self.results.index, self.results['hi1_close'],
                 label='HI1 close', color='blue', linewidth=1.2)
        ax1.set_ylabel('price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # 权益曲线
        ax2 = ax1.twinx()
        ax2.plot(self.results.index, self.results['equity'],
                 label='capital', color='red', linewidth=2)
        ax2.set_ylabel('equity (HKD)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f'{self.strategy_name} - 回测结果')
        fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 读取清洗后的数据
    df = pd.read_csv(CLEANED_FILE, parse_dates=['datetime'], index_col='datetime')
    print(f"加载数据完成，共 {len(df):,} 条记录")
    print(f"时间范围: {df.index.min()} → {df.index.max()}")

    # ========== 示例1：BuyAndHold，关闭日内平仓开关 ==========
    print("\n" + "=" * 60)
    print("回测 BuyAndHold（允许隔夜持仓）")
    print("=" * 60)
    strategy_bh = BuyAndHoldStrategy()
    backtester_bh = AdvancedBacktester(
        data=df,
        initial_capital=INITIAL_CAPITAL,
        multiplier=MULTIPLIER,
        commission_per_trade=COMMISSION_PER_TRADE,
        slippage_points=1,
        force_intraday=False      # ← 关键：不强制平仓
    )
    results_bh = backtester_bh.run(strategy_bh)
    backtester_bh.plot()
    results_bh.to_csv('backtest_buy_and_hold_results.csv')
    print("\nBuyAndHold 结果已保存：backtest_buy_and_hold_results.csv")

    # ========== 示例2：一个占位的日内策略（强制平仓） ==========
    # 你可以将自己的日内策略类放在这里测试，并开启 force_intraday=True
    # print("\n" + "=" * 60)
    # print("回测日内策略（强制平仓）")
    # print("=" * 60)
    # strategy_intraday = YourIntradayStrategy()
    # backtester_intra = AdvancedBacktester(
    #     data=df,
    #     initial_capital=INITIAL_CAPITAL,
    #     multiplier=MULTIPLIER,
    #     commission_per_trade=COMMISSION_PER_TRADE,
    #     slippage_points=1,
    #     force_intraday=True      # 日内策略强制尾盘平仓
    # )
    # results_intra = backtester_intra.run(strategy_intraday)
    # backtester_intra.plot()
    # results_intra.to_csv('backtest_intraday_results.csv')