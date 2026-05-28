# 三篇样例论文写作风格与图片类型总结

本文基于 `exmaple` 文件夹中三篇 STAT8020 项目 PDF 的正文结构、图注、表格标题和附录代码线索进行总结，重点关注它们的写作风格、图表使用方式，以及对本项目 final report 可借鉴的写法。

## 1. `project sample1.pdf`

### 主题与结构

该报告主题是基于 VIX 与 S&P 500 负相关关系的 contrarian strategy。整体结构非常接近课程项目报告模板：

- Introduction
- Proposed strategy
- Backtesting
- Out-sample testing
- Observations and discussions
- Conclusion
- Reference
- Appendix

报告先用半强式有效市场假说引入，再用 behavioral finance 中的 contrarian investing 说明为什么市场恐慌时可能存在买入机会。随后提出明确交易规则：当 VIX 高于阈值时买入 SPX futures，当 VIX 连续低于阈值若干天后卖出。

### 写作风格

这篇报告偏“故事驱动 + 规则解释型”。它先解释经济直觉，再给出简单阈值规则，之后用 backtesting 和 out-sample testing 检验。语言比较直接，适合课程项目：每个模型假设、参数含义和结果解释都写得比较清楚。

主要特点：

- Introduction 中先引用 EMH，再说明现实市场中投资者并不总是理性，因此 contrarian strategy 可能有效。
- Rationale 写得较自然，强调“market panic”“cheap price”“risk-taking mode”等经济解释。
- 参数选择部分不是只选最大利润，而会解释为什么极端阈值虽然训练集表现好但不一定合理。
- Out-sample testing 后会主动讨论失败或无法评估的参数组合，例如没有触发卖出信号。
- Discussion 部分很实用，会讨论市场波动结构变化、持仓周期、参数主观性、流动性风险和 futures rollover。

### 图片和表格类型

该报告主要使用以下类型的可视化：

- 双轴时间序列图：VIX futures 与 SPX futures 同图展示，用来说明 VIX 与股票指数的负相关关系。
- Out-sample 时间序列图：单独展示测试期 VIX 与 SPX 的走势，解释为什么样本外表现不同。
- 局部时期放大图：例如 2008-2010 年 VIX 与 SPX，用于解释极端波动环境。
- 参数选择表：不同 quiet days holding period 下的 maximum profit。
- 参数组合表现表：比较不同阈值和持有期的 cumulative profits。
- 交易日期表：列出 purchase day、purchase price、sold day、sold price 和 profit。
- 附录代码：R 代码，用于展示策略回测实现。

### 可借鉴点

- Introduction 可以先讲 EMH，再自然转入市场异常和策略动机。
- 参数选择不应只写“最大值”，还应解释为什么某些最大值可能是过拟合或不具有交易意义。
- Out-sample 结果如果与 validation/backtesting 不一致，应主动解释 regime change。
- 实盘讨论可以加入 liquidity risk、forced close、rollover、transaction cost 等现实限制。

## 2. `project sample2.pdf`

### 主题与结构

该报告研究 Home Depot 股票上的 momentum 和 mean-reversion 组合策略。核心方法是将 Z-score mean-reversion signal 与 EWMA momentum signal 组合，并通过平滑参数得到最终 trade position。

结构如下：

- Introduction
- Proposed strategy
  - Rationale
  - Trading algorithm
- Backtesting
  - Parameter selection
  - Back-testing result
- Out-sample testing
- Observations and discussions
  - Remove mean-reversion position
  - Add trend signal
- Conclusion
- Reference
- Appendix

### 写作风格

这篇报告偏“公式定义 + 参数实验型”。它的策略相对更技术化，因此在 Proposed strategy 中较多使用数学表达式和分步骤算法。

主要特点：

- Introduction 用 EMH、momentum anomaly、contrarian anomaly 引出策略组合。
- Rationale 直接解释为什么 EWMA 用于 momentum，Z-score 用于 mean reversion。
- Trading Algorithm 用编号步骤定义信号生成流程，便于读者复现。
- Backtesting 部分集中比较不同 rolling window、EWMA decay factor、signal decay 参数。
- Out-sample testing 发现策略无法打败 BAH 后，没有回避问题，而是解释 bull market 中 equal weight 的 momentum/mean-reversion 组合不适合。
- Discussion 不是泛泛而谈，而是做了两个方向的改进实验：移除 mean-reversion position 和加入 trend signal。

### 图片和表格类型

该报告主要使用以下类型的图：

- 标的价格时间序列图：HD close price，用于说明样本期间市场状态。
- 参数敏感性回测图：不同 Z-score rolling lag 的累计收益曲线。
- EWMA 参数比较图：不同 decay factor 下的策略累计收益。
- Signal decay 参数比较图：不同 smoothing parameter 下的策略累计收益。
- Strategy vs Buy-and-Hold 对比图：训练集和测试集分别展示。
- 改进实验图：移除 Z-score 后的策略表现图。
- 附录代码：Python 代码，使用 pandas、numpy、matplotlib、seaborn 实现策略。

该报告表格较少，更多依赖多条折线图进行参数比较。

### 可借鉴点

- 对模型细节的写法可以采用“公式 + 编号步骤 + 参数解释”的方式。
- 参数优化部分适合用多组曲线说明，而不只是给最终最佳参数。
- 如果 out-of-sample 表现不好，应明确说明失败原因，并提出有针对性的改进方向。
- Discussion 可以围绕“为什么该策略在某种市场状态下失效”展开，而不是只重复结果。

## 3. `project sample3.pdf`

### 主题与结构

该报告比较了三套基于 Hang Seng Index 的策略：

- Kelly-Inspired Strategy
- Filter Trading Rule
- Momentum Strategy based on SMA and Bollinger Bands

结构比前两篇更像多策略综合报告：

- Introduction
- Kelly-Inspired Strategy
- Filter Trading Rule
- Momentum Back-testing Strategy
- Observations and discussions
- Conclusion
- Appendix

每个策略内部又包含 introduction、methodology、parameter selection、back testing、out-sample test 等小节。

### 写作风格

这篇报告偏“多策略横向比较型”。它不是深入一个模型，而是分别介绍三套策略，然后在 observations、risk management 和 conclusion 中进行比较。

主要特点：

- Introduction 中先介绍 EMH、random walk、Kelly formula、filter rule 和 SMA 等背景，再说明比较多套策略的目的。
- 每个策略都独立成章，先讲 intuition，再讲 methodology，然后给 backtesting 和 out-sample testing。
- 对 Kelly strategy 使用流程图解释策略运行过程，这比纯文字更直观。
- Filter trading rule 和 Momentum strategy 都强调 parameter selection，并把最优参数用于 out-sample。
- Discussion 额外加入 slippage、implementation difficulty、risk management、variance 和 VaR，覆盖课程 report 要求较完整。
- Conclusion 按策略逐一总结，而不是只写一个笼统结论。

### 图片和表格类型

该报告使用的图表类型最丰富：

- 标的价格走势图：Hang Seng Index adjusted close price。
- 策略流程图：Kelly-Inspired strategy 的 p、R、theoretical wealth、position adjustment 和 wealth calculation 流程。
- BAH vs strategy 累计财富图：比较 Kelly、filter rule、momentum strategy 与 buy-and-hold。
- 参数选择折线图：例如 delta 与 sum strategy return 的关系。
- 多策略累计收益对比图：短期、中期、长期 Bollinger momentum scenarios 与 BAH。
- Out-sample 对比图：用 HSBC 作为样本外标的检验。
- Slippage 敏感性图：不同 slippage 水平下累计收益变化。
- Trading position/frequency 图：展示交易频率和仓位变化。
- 参数选择表：短中长期 SMA 与 Bollinger band multiplier。
- Slippage 表：不同 slippage 对 cumulative return 的影响。
- Risk measurement 表：variance 和 VaR。
- 附录截图/代码：包含 notebook 输出、数据表、Python 代码和图像输出。

### 可借鉴点

- 当报告包含三套策略时，可以采用“每套策略独立介绍 + 后面统一比较”的结构。
- 图片类型不应只放收益曲线，还可以加入流程图、参数敏感性图、slippage 图和风险指标表。
- 风险管理章节可以用 variance、VaR、drawdown、trading frequency 等指标支撑。
- Conclusion 可以按策略分段总结，便于读者快速看清每套策略的优缺点。

## 三篇样例的共同写作规律

### 共同结构

三篇报告基本都遵循以下逻辑：

1. 先用 EMH 或市场异常说明为什么策略可能有意义。
2. 再介绍数据来源和标的资产。
3. 然后定义策略规则或模型公式。
4. 接着进行参数选择或 backtesting。
5. 再做 out-sample testing。
6. 最后讨论结果、实盘问题、风险管理和结论。

这说明课程报告最看重的不是单纯展示最终收益，而是完整说明从理论动机到样本外检验的过程。

### 写作语气

三篇样例都使用较直接的课程项目风格：

- 理论介绍不长，主要服务于策略动机。
- 策略规则写得具体，尽量能复现。
- 对结果的解释通常结合市场状态，而不是只列数字。
- 样本外表现不好时，会给出原因和改进方向。
- 结论会回到 buy-and-hold benchmark、参数稳健性和实盘限制。

### 常见图片类型

可归纳为七类：

1. 标的价格走势图：展示研究对象的市场背景。
2. 策略变量关系图：例如 VIX vs SPX、价格与指标的关系。
3. 参数敏感性图：比较不同 window、threshold、decay factor、delta、slippage。
4. 策略 vs BAH 累计收益/累计财富图：最核心的表现对比图。
5. Out-sample 测试图：展示样本外策略表现。
6. 风险管理图：仓位、交易频率、slippage sensitivity、drawdown 或 VaR。
7. 策略流程图：用于解释较复杂的交易逻辑。

### 常见表格类型

常见表格包括：

- 参数网格和参数说明表。
- 不同参数组合的收益表现表。
- 样本内/样本外 performance comparison 表。
- 交易记录表，包括 entry、exit、price、profit。
- Slippage sensitivity 表。
- Risk measurement 表，包括 variance、VaR、drawdown 或 win rate。

## 对本项目报告的建议

结合三篇样例，本项目 final report 可以重点模仿以下写法：

1. Introduction 采用样例 1 和样例 2 的方式：先讲 EMH，再讲市场异常、波动聚集、momentum/mean reversion 和机器学习策略动机。
2. 三套策略结构采用样例 3 的方式：LGBM、GRPO、GARCH-Bollinger 分别介绍，但在回测和风险管理章节统一比较。
3. 模型细节采用样例 2 的方式：用公式和编号步骤解释信号、目标函数和参数。
4. 参数优化采用样例 1 和样例 2 的方式：不仅写最佳参数，还说明为什么 validation-only selection 可以避免 data snooping。
5. 回测结果采用样例 3 的方式：每套策略给 benchmark、测试集表现、风险指标和实盘限制。
6. 图片可以优先准备以下几类：
   - HSI futures price 或 cumulative return 走势图。
   - 三套策略的 equity curve / cumulative return 对比图。
   - 参数敏感性或 validation selection 图。
   - Slippage 或 transaction cost sensitivity 图。
   - Drawdown 或 trading position 图。
   - GRPO 或整体实验流程图。

如果报告篇幅有限，最值得保留的是 cumulative return/equity curve、drawdown、参数选择结果、slippage sensitivity 和策略流程图。这些图最能对应课程要求中的 backtesting、optimization、slippage、implementation difficulty 和 risk management。
