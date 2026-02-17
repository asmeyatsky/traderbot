import { formatCurrency, formatPercent } from '../../lib/format';
import type { PerformanceItem } from '../../types/dashboard';

interface TopMoversProps {
  gainers: PerformanceItem[];
  losers: PerformanceItem[];
}

export default function TopMovers({ gainers, losers }: TopMoversProps) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
        <h3 className="text-sm font-medium text-gray-500">Top Gainers</h3>
        <ul className="mt-3 divide-y divide-gray-100">
          {gainers.map((item) => (
            <li key={item.symbol} className="flex items-center justify-between py-2">
              <span className="font-medium text-gray-900">{item.symbol}</span>
              <div className="text-right">
                <span className="text-sm text-gray-500">{formatCurrency(item.current_price)}</span>
                <span className="ml-2 text-sm font-medium text-green-600">
                  {formatPercent(item.change_percent)}
                </span>
              </div>
            </li>
          ))}
          {gainers.length === 0 && <li className="py-2 text-sm text-gray-400">No data</li>}
        </ul>
      </div>
      <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
        <h3 className="text-sm font-medium text-gray-500">Top Losers</h3>
        <ul className="mt-3 divide-y divide-gray-100">
          {losers.map((item) => (
            <li key={item.symbol} className="flex items-center justify-between py-2">
              <span className="font-medium text-gray-900">{item.symbol}</span>
              <div className="text-right">
                <span className="text-sm text-gray-500">{formatCurrency(item.current_price)}</span>
                <span className="ml-2 text-sm font-medium text-red-600">
                  {formatPercent(item.change_percent)}
                </span>
              </div>
            </li>
          ))}
          {losers.length === 0 && <li className="py-2 text-sm text-gray-400">No data</li>}
        </ul>
      </div>
    </div>
  );
}
