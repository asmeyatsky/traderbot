/**
 * Centralized Help Text Constants
 *
 * Architectural Intent:
 * Single source of truth for all contextual help text displayed throughout
 * the application. Keeps UI components focused on rendering while help
 * content is maintained in one place.
 */

export const ORDER_TYPE_HELP: Record<string, string> = {
  MARKET: 'Executes immediately at the current market price. Fast but price may vary.',
  LIMIT: 'Executes only at your specified price or better. May not fill if the price is not reached.',
  STOP_LOSS: 'Triggers a market order when the price reaches your stop price. Used to limit losses.',
  TRAILING_STOP: 'A stop order that trails the price by a fixed amount, locking in gains as price moves favorably.',
};

export const ORDER_SIDE_HELP: Record<string, string> = {
  BUY: 'Purchase shares of a stock. You profit when the price goes up.',
  SELL: 'Sell shares you own. Closes or reduces an existing position.',
};

export const RISK_METRIC_HELP: Record<string, string> = {
  'VaR (95%)': 'Value at Risk: the maximum expected loss over a day with 95% confidence. A VaR of -2% means you have a 5% chance of losing more than 2% in a day.',
  'VaR (99%)': 'Value at Risk at 99% confidence. Stricter measure — only a 1% chance of exceeding this loss.',
  'Expected Shortfall': 'Average loss in the worst-case scenarios beyond VaR. Shows how bad losses could get in extreme conditions.',
  'Max Drawdown': 'The largest peak-to-trough decline in portfolio value. Measures the worst historical loss from a high point.',
  'Volatility': 'How much your portfolio value fluctuates. Higher volatility means more unpredictable returns.',
  'Beta': 'How your portfolio moves relative to the market. Beta > 1 means more volatile than the market; < 1 means less.',
  'Sharpe Ratio': 'Risk-adjusted return. Higher is better — shows how much return you earn per unit of risk. Above 1.0 is generally good.',
  'Sortino Ratio': 'Like Sharpe but only penalizes downside volatility. Higher is better — focuses on harmful risk only.',
};

export const INDICATOR_HELP: Record<string, string> = {
  'RSI (14)': 'Relative Strength Index: measures momentum on a 0-100 scale. Above 70 = overbought (may fall), below 30 = oversold (may rise).',
  'MACD': 'Moving Average Convergence Divergence: shows trend direction and momentum. Positive = bullish trend.',
  'MACD Signal': 'The signal line for MACD. When MACD crosses above this line, it suggests a buy signal.',
  'SMA 20': 'Simple Moving Average over 20 days. Smooths out short-term price fluctuations.',
  'SMA 50': 'Simple Moving Average over 50 days. Shows the medium-term trend direction.',
  'EMA 12': 'Exponential Moving Average (12-day). Reacts faster to recent price changes than SMA.',
  'EMA 26': 'Exponential Moving Average (26-day). Used with EMA 12 to calculate MACD.',
  'Bollinger Upper': 'Upper Bollinger Band. Price near this level may indicate the stock is overbought.',
  'Bollinger Lower': 'Lower Bollinger Band. Price near this level may indicate the stock is oversold.',
  'ATR': 'Average True Range: measures daily price volatility. Higher ATR = more volatile stock.',
};

export const ML_HELP = {
  confidence: 'How confident the ML model is in its prediction. Higher percentage = stronger conviction.',
  score: 'Composite score combining multiple model outputs. Ranges from -1 (strong sell) to +1 (strong buy).',
  signal: 'Trading signal derived from ML analysis, news sentiment, and technical indicators combined.',
  news_impact: 'How recent news is expected to affect the stock price, based on sentiment analysis.',
};

export const STRESS_TEST_HELP = {
  impact: 'Estimated change in portfolio value if this scenario occurs.',
  probability: 'Estimated likelihood of this scenario happening based on historical data.',
};

export const RISK_TOLERANCE_HELP: Record<string, string> = {
  CONSERVATIVE: 'Lower risk, steadier returns. Prioritizes capital preservation with smaller position sizes and tighter stop-losses.',
  MODERATE: 'Balanced approach. Accepts moderate fluctuations for reasonable growth potential.',
  AGGRESSIVE: 'Higher risk for higher potential returns. Larger positions, wider stop-losses, and more exposure to volatile stocks.',
};

export const INVESTMENT_GOAL_HELP: Record<string, string> = {
  CAPITAL_PRESERVATION: 'Focus on protecting your initial investment. Best for short time horizons or low risk tolerance.',
  BALANCED_GROWTH: 'Grow your portfolio steadily while managing risk. Good for medium-term goals.',
  MAXIMUM_RETURNS: 'Maximize growth potential. Accepts higher volatility and risk for the best possible returns.',
};

export const PAGE_DESCRIPTIONS: Record<string, string> = {
  dashboard: 'Overview of your portfolio performance, auto-trading status, and recent activity.',
  trading: 'Place buy and sell orders, and track your order history.',
  portfolio: 'View your holdings, cash balance, and portfolio allocation.',
  market_data: 'Look up real-time prices, charts, and technical indicators for any stock.',
  predictions: 'AI-powered price predictions and trading signals based on machine learning models.',
  risk: 'Portfolio risk metrics and stress test scenarios to understand your exposure.',
  activity: 'Full history of auto-trading signals, orders, and executions.',
  settings: 'Manage your profile, risk preferences, and auto-trading configuration.',
};

export const AUTO_TRADING_HELP =
  'When enabled, TraderBot automatically generates trading signals using ML models and places orders on your watchlist stocks within your risk parameters.';
