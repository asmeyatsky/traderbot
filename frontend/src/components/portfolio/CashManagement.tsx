import { useState } from 'react';
import { formatCurrency } from '../../lib/format';
import { useDepositCash, useWithdrawCash } from '../../hooks/use-portfolio';

interface CashManagementProps {
  cashBalance: number;
  totalValue: number;
}

const PRESETS = [1000, 5000, 10000];

export default function CashManagement({ cashBalance, totalValue }: CashManagementProps) {
  const cashPercent = totalValue > 0 ? (cashBalance / totalValue) * 100 : 0;
  const [action, setAction] = useState<'deposit' | 'withdraw' | null>(null);
  const [amount, setAmount] = useState('');
  const deposit = useDepositCash();
  const withdraw = useWithdrawCash();

  const isPending = deposit.isPending || withdraw.isPending;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const value = Number(amount);
    if (value <= 0) return;
    if (action === 'deposit') {
      deposit.mutate(value, { onSuccess: () => reset() });
    } else if (action === 'withdraw') {
      withdraw.mutate(value, { onSuccess: () => reset() });
    }
  }

  function reset() {
    setAction(null);
    setAmount('');
  }

  function selectPreset(value: number) {
    setAmount(String(value));
  }

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
      <p className="mt-2 text-xs text-gray-400">This is virtual money for paper trading.</p>

      {!action ? (
        <div className="mt-4 flex gap-2">
          <button
            type="button"
            onClick={() => setAction('deposit')}
            className="flex-1 rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700"
          >
            Deposit
          </button>
          <button
            type="button"
            onClick={() => setAction('withdraw')}
            disabled={cashBalance <= 0}
            className="flex-1 rounded-md bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm ring-1 ring-gray-300 hover:bg-gray-50 disabled:opacity-50"
          >
            Withdraw
          </button>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="mt-4 space-y-3">
          <p className="text-sm font-medium text-gray-700">
            {action === 'deposit' ? 'Add virtual funds' : 'Withdraw funds'}
          </p>

          {action === 'deposit' && (
            <div className="flex gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => selectPreset(p)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                    amount === String(p)
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {formatCurrency(p)}
                </button>
              ))}
            </div>
          )}

          <div className="relative">
            <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">$</span>
            <input
              type="number"
              min={1}
              max={action === 'withdraw' ? cashBalance : undefined}
              step="0.01"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="Amount"
              required
              className="block w-full rounded-md border border-gray-300 py-2 pl-7 pr-3 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>

          {withdraw.error && action === 'withdraw' && (
            <p className="text-xs text-red-600">Insufficient balance.</p>
          )}
          {deposit.error && action === 'deposit' && (
            <p className="text-xs text-red-600">Deposit failed. Try again.</p>
          )}

          <div className="flex gap-2">
            <button
              type="button"
              onClick={reset}
              disabled={isPending}
              className="flex-1 rounded-md bg-white px-3 py-1.5 text-sm font-medium text-gray-700 ring-1 ring-gray-300 hover:bg-gray-50 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending || Number(amount) <= 0}
              className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium text-white shadow-sm disabled:opacity-50 ${
                action === 'deposit'
                  ? 'bg-indigo-600 hover:bg-indigo-700'
                  : 'bg-amber-600 hover:bg-amber-700'
              }`}
            >
              {isPending ? 'Processing...' : action === 'deposit' ? 'Deposit' : 'Withdraw'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
