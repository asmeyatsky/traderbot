import type { Signal } from '../../types/ml';

const signalColors: Record<string, string> = {
  STRONG_BUY: 'text-green-700 bg-green-100',
  BUY: 'text-green-600 bg-green-50',
  HOLD: 'text-yellow-700 bg-yellow-100',
  SELL: 'text-red-600 bg-red-50',
  STRONG_SELL: 'text-red-700 bg-red-100',
};

export default function SignalGauge({ data }: { data: Signal }) {
  const color = signalColors[data.signal] ?? 'text-gray-700 bg-gray-100';
  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-sm font-medium text-gray-500">{data.symbol} Signal</h3>
      <div className={`mt-3 inline-flex rounded-full px-3 py-1 text-sm font-bold ${color}`}>
        {data.signal}
      </div>
      <p className="mt-2 text-sm text-gray-600">
        Confidence: {(data.confidence * 100).toFixed(0)}%
      </p>
      <p className="mt-1 text-xs text-gray-500">News impact: {data.news_impact}</p>
    </div>
  );
}
