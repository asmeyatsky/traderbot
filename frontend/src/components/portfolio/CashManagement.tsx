import { formatCurrency } from '../../lib/format';

interface CashManagementProps {
  cashBalance: number;
  totalValue: number;
}

export default function CashManagement({ cashBalance, totalValue }: CashManagementProps) {
  const cashPercent = totalValue > 0 ? (cashBalance / totalValue) * 100 : 0;

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-sm font-medium text-gray-500">Cash Balance</h3>
      <p className="mt-2 text-2xl font-semibold text-gray-900">{formatCurrency(cashBalance)}</p>
      <p className="mt-1 text-sm text-gray-500">{cashPercent.toFixed(1)}% of portfolio</p>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-200">
        <div
          className="h-full rounded-full bg-indigo-600"
          style={{ width: `${Math.min(cashPercent, 100)}%` }}
        />
      </div>
    </div>
  );
}
