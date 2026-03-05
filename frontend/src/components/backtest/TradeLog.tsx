import type { BacktestTrade } from '../../types/backtest';

interface Props {
  trades: BacktestTrade[];
}

export default function TradeLog({ trades }: Props) {
  if (trades.length === 0) {
    return <p className="text-sm text-gray-500 dark:text-gray-400">No trades executed.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-sm">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
            <th className="pb-2 pr-3 font-medium">Date</th>
            <th className="pb-2 pr-3 font-medium">Action</th>
            <th className="pb-2 pr-3 font-medium">Symbol</th>
            <th className="pb-2 pr-3 font-medium text-right">Qty</th>
            <th className="pb-2 pr-3 font-medium text-right">Price</th>
            <th className="pb-2 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.slice(0, 50).map((t, i) => (
            <tr key={i} className="border-b border-gray-100 dark:border-gray-700">
              <td className="py-1.5 pr-3 text-gray-600 dark:text-gray-400">
                {t.date ? String(t.date).slice(0, 10) : '-'}
              </td>
              <td className={`py-1.5 pr-3 font-medium ${t.action === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>
                {t.action}
              </td>
              <td className="py-1.5 pr-3 font-semibold text-gray-900 dark:text-white">{t.symbol}</td>
              <td className="py-1.5 pr-3 text-right text-gray-700 dark:text-gray-300">{t.quantity}</td>
              <td className="py-1.5 pr-3 text-right text-gray-700 dark:text-gray-300">${t.price?.toFixed(2)}</td>
              <td className="py-1.5 text-gray-500 dark:text-gray-400 truncate max-w-[200px]">{t.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
