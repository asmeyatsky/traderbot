import DataTable from '../common/DataTable';
import { formatCurrency, formatPercent } from '../../lib/format';
import type { Position } from '../../types/portfolio';

interface PositionsTableProps {
  positions: Position[];
  onTrade?: (symbol: string) => void;
}

export default function PositionsTable({ positions, onTrade }: PositionsTableProps) {
  return (
    <DataTable
      data={positions}
      keyExtractor={(p) => p.symbol}
      emptyMessage="No positions"
      columns={[
        { key: 'symbol', header: 'Symbol', render: (p) => <span className="font-medium">{p.symbol}</span> },
        { key: 'qty', header: 'Qty', render: (p) => p.quantity },
        { key: 'avgCost', header: 'Avg Cost', render: (p) => formatCurrency(p.average_buy_price) },
        { key: 'price', header: 'Price', render: (p) => formatCurrency(p.current_price) },
        { key: 'value', header: 'Market Value', render: (p) => formatCurrency(p.market_value) },
        {
          key: 'pnl',
          header: 'P&L',
          render: (p) => (
            <span className={p.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
              {formatCurrency(p.unrealized_pnl)} ({formatPercent(p.pnl_percentage)})
            </span>
          ),
        },
        {
          key: 'today',
          header: 'Today',
          render: (p) => {
            const change = p.day_change ?? 0;
            const pct = p.day_change_percent ?? 0;
            const color = change >= 0 ? 'text-green-600' : 'text-red-600';
            return (
              <span className={color}>
                {change >= 0 ? '+' : ''}{formatCurrency(change)} ({change >= 0 ? '+' : ''}{formatPercent(pct)})
              </span>
            );
          },
        },
        ...(onTrade
          ? [{
              key: 'trade',
              header: '',
              render: (p: Position) => (
                <button
                  onClick={() => onTrade(p.symbol)}
                  className="rounded-md bg-indigo-50 dark:bg-indigo-900/30 px-2.5 py-1 text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-900/50 transition-colors"
                >
                  Trade
                </button>
              ),
            }]
          : []),
      ]}
    />
  );
}
