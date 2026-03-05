import type { TradeAction } from '../../types/chat';

interface TradeActionCardProps {
  action: TradeAction;
  onConfirm?: () => void;
  isExecuting?: boolean;
  isExecuted?: boolean;
}

export default function TradeActionCard({ action, onConfirm, isExecuting, isExecuted }: TradeActionCardProps) {
  const isBuy = action.action === 'BUY';
  const executed = isExecuted || action.executed;

  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 p-3 dark:border-gray-600 dark:bg-gray-700/50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`rounded-md px-2 py-0.5 text-xs font-bold ${
              isBuy
                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-400'
                : 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-400'
            }`}
          >
            {action.action}
          </span>
          <span className="font-semibold text-gray-900 dark:text-white">{action.symbol}</span>
          <span className="text-gray-500 dark:text-gray-400">{action.quantity} shares</span>
        </div>
      </div>

      {action.reasoning && (
        <p className="mt-1.5 text-xs text-gray-600 dark:text-gray-400">{action.reasoning}</p>
      )}

      {!executed && !isExecuting && (
        <button
          onClick={onConfirm}
          className={`mt-2 w-full rounded-lg py-1.5 text-xs font-medium text-white transition-colors ${
            isBuy
              ? 'bg-emerald-600 hover:bg-emerald-700'
              : 'bg-red-600 hover:bg-red-700'
          }`}
        >
          Confirm {action.action}
        </button>
      )}

      {isExecuting && (
        <div className="mt-2 flex items-center justify-center gap-2 rounded-lg bg-gray-200 py-1.5 text-xs font-medium text-gray-600 dark:bg-gray-600 dark:text-gray-300">
          <svg className="h-3 w-3 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Placing Order...
        </div>
      )}

      {executed && !isExecuting && (
        <div className="mt-2 rounded-lg bg-emerald-100 py-1.5 text-center text-xs font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
          Order Executed
        </div>
      )}
    </div>
  );
}
