import type { TechnicalIndicators } from '../../types/dashboard';
import { formatNumber } from '../../lib/format';
import { INDICATOR_HELP } from '../../lib/help-text';
import InfoTooltip from '../common/InfoTooltip';

export default function IndicatorPanel({ data }: { data: TechnicalIndicators }) {
  const indicators = [
    { label: 'RSI (14)', value: formatNumber(data.rsi), signal: data.rsi > 70 ? 'Overbought' : data.rsi < 30 ? 'Oversold' : 'Neutral' },
    { label: 'MACD', value: formatNumber(data.macd, 4) },
    { label: 'MACD Signal', value: formatNumber(data.macd_signal, 4) },
    { label: 'SMA 20', value: formatNumber(data.sma_20) },
    { label: 'SMA 50', value: formatNumber(data.sma_50) },
    { label: 'EMA 12', value: formatNumber(data.ema_12) },
    { label: 'EMA 26', value: formatNumber(data.ema_26) },
    { label: 'Bollinger Upper', value: formatNumber(data.bollinger_upper) },
    { label: 'Bollinger Lower', value: formatNumber(data.bollinger_lower) },
    { label: 'ATR', value: formatNumber(data.atr) },
  ];

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-sm font-medium text-gray-500">Technical Indicators</h3>
      <p className="mt-1 text-xs text-gray-400">Technical indicators help identify trends and trading opportunities.</p>
      <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {indicators.map((ind) => (
          <div key={ind.label} className="rounded-md bg-gray-50 px-3 py-2">
            <dt className="flex items-center gap-1 text-xs text-gray-500">
              {ind.label}
              {INDICATOR_HELP[ind.label] && <InfoTooltip text={INDICATOR_HELP[ind.label]} />}
            </dt>
            <dd className="mt-0.5 text-sm font-medium text-gray-900">{ind.value}</dd>
            {'signal' in ind && ind.signal && (
              <dd className={`text-xs ${ind.signal === 'Overbought' ? 'text-red-600' : ind.signal === 'Oversold' ? 'text-green-600' : 'text-gray-500'}`}>
                {ind.signal}
              </dd>
            )}
          </div>
        ))}
      </dl>
    </div>
  );
}
