import { useState } from 'react';
import { useStrategies, useRunBacktest } from '../hooks/use-backtest';
import EquityCurve from '../components/backtest/EquityCurve';
import TradeLog from '../components/backtest/TradeLog';

export default function BacktestPage() {
  const { data: strategiesData } = useStrategies();
  const mutation = useRunBacktest();

  const [strategy, setStrategy] = useState('sma_crossover');
  const [symbol, setSymbol] = useState('AAPL');
  const [capital, setCapital] = useState('10000');

  const strategies = strategiesData?.strategies ?? [];
  const result = mutation.data;

  function handleRun() {
    mutation.mutate({
      strategy,
      symbol: symbol.toUpperCase(),
      initial_capital: parseFloat(capital) || 10000,
    });
  }

  return (
    <div className="h-full overflow-auto p-4 sm:p-6">
      <h1 className="text-lg font-semibold text-gray-800">Backtesting</h1>
      <p className="mt-1 text-sm text-gray-500">
        Test trading strategies against historical data.
      </p>

      {/* Config form */}
      <div className="mt-4 flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs font-medium text-gray-600">Strategy</label>
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
            className="mt-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
          >
            {strategies.map((s) => (
              <option key={s.name} value={s.name}>
                {s.label}
              </option>
            ))}
            {strategies.length === 0 && (
              <>
                <option value="sma_crossover">SMA Crossover</option>
                <option value="rsi_mean_reversion">RSI Mean Reversion</option>
                <option value="momentum">Momentum Breakout</option>
              </>
            )}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600">Symbol</label>
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="mt-1 w-24 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-gray-600">Capital ($)</label>
          <input
            value={capital}
            onChange={(e) => setCapital(e.target.value)}
            type="number"
            className="mt-1 w-28 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
          />
        </div>

        <button
          onClick={handleRun}
          disabled={mutation.isPending}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {mutation.isPending ? 'Running...' : 'Run Backtest'}
        </button>
      </div>

      {/* Results */}
      {result && !result.error && (
        <div className="mt-6 space-y-6">
          {/* Metrics */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric label="Final Value" value={`$${result.final_value.toLocaleString()}`} />
            <Metric label="Total Return" value={`${result.total_return_pct}%`} positive={result.total_return_pct >= 0} />
            <Metric label="Sharpe Ratio" value={result.sharpe_ratio.toFixed(2)} />
            <Metric label="Max Drawdown" value={`${result.max_drawdown_pct}%`} />
            <Metric label="Win Rate" value={`${result.win_rate}%`} />
            <Metric label="Total Trades" value={String(result.total_trades)} />
            <Metric label="Volatility" value={`${result.volatility}%`} />
            <Metric label="Profit Factor" value={result.profit_factor.toFixed(2)} />
          </div>

          {/* Equity Curve */}
          <div>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Equity Curve</h2>
            <EquityCurve trades={result.trades} initialCapital={result.initial_capital} />
          </div>

          {/* Trade Log */}
          <div>
            <h2 className="mb-2 text-sm font-semibold text-gray-700">Trade Log</h2>
            <TradeLog trades={result.trades} />
          </div>
        </div>
      )}

      {result?.error && (
        <div className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{result.error}</div>
      )}
    </div>
  );
}

function Metric({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className={`mt-0.5 text-lg font-semibold ${positive === true ? 'text-green-600' : positive === false ? 'text-red-600' : 'text-gray-900'}`}>
        {value}
      </p>
    </div>
  );
}
