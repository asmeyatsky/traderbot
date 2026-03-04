import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { BacktestTrade } from '../../types/backtest';

interface Props {
  trades: BacktestTrade[];
  initialCapital: number;
}

export default function EquityCurve({ trades, initialCapital }: Props) {
  // Build equity curve from trades
  const points = [{ date: 'Start', value: initialCapital }];
  let cash = initialCapital;

  for (const t of trades) {
    if (t.action === 'BUY') {
      cash -= t.value + t.commission;
    } else if (t.action === 'SELL') {
      cash += t.value - t.commission;
    }
    points.push({
      date: t.date ? t.date.slice(0, 10) : `Trade ${points.length}`,
      value: Math.round(cash * 100) / 100,
    });
  }

  if (points.length < 2) {
    return <p className="text-sm text-gray-500">No trades to display.</p>;
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10 }} tickFormatter={(v: number) => `$${v.toLocaleString()}`} />
          <Tooltip formatter={(v) => [`$${Number(v).toLocaleString()}`, 'Portfolio']} />
          <Line type="monotone" dataKey="value" stroke="#4f46e5" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
