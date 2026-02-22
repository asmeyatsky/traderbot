import type { RiskMetrics } from '../../types/risk';
import { formatPercent, formatNumber } from '../../lib/format';
import { RISK_METRIC_HELP } from '../../lib/help-text';
import InfoTooltip from '../common/InfoTooltip';

export default function RiskMetricsCard({ data }: { data: RiskMetrics }) {
  const isEmpty =
    data.var_95 === 0 && data.var_99 === 0 && data.volatility === 0 && data.beta === 0;

  const metrics = [
    { label: 'VaR (95%)', value: formatPercent(data.var_95 * 100) },
    { label: 'VaR (99%)', value: formatPercent(data.var_99 * 100) },
    { label: 'Expected Shortfall', value: formatPercent(data.expected_shortfall * 100) },
    { label: 'Max Drawdown', value: formatPercent(data.max_drawdown * 100) },
    { label: 'Volatility', value: formatPercent(data.volatility * 100) },
    { label: 'Beta', value: formatNumber(data.beta) },
    { label: 'Sharpe Ratio', value: formatNumber(data.sharpe_ratio) },
    { label: 'Sortino Ratio', value: formatNumber(data.sortino_ratio) },
  ];

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-sm font-medium text-gray-500">Risk Metrics</h3>
      {isEmpty ? (
        <p className="mt-4 text-sm text-gray-400">
          Add positions to your portfolio to see risk analytics. Metrics are calculated based on your holdings.
        </p>
      ) : (
        <dl className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          {metrics.map((m) => (
            <div key={m.label}>
              <dt className="flex items-center gap-1 text-xs text-gray-500">
                {m.label}
                {RISK_METRIC_HELP[m.label] && <InfoTooltip text={RISK_METRIC_HELP[m.label]} />}
              </dt>
              <dd className="mt-0.5 text-lg font-semibold text-gray-900">{m.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
