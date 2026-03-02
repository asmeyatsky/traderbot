import type { TradeAction } from '../../types/chat';

interface TradeActionCardProps {
  action: TradeAction;
  onConfirm?: () => void;
}

export default function TradeActionCard({ action, onConfirm }: TradeActionCardProps) {
  const isBuy = action.action === 'BUY';

  return (
    <div className="rounded-xl border border-gray-200 bg-gray-50 p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`rounded-md px-2 py-0.5 text-xs font-bold ${
              isBuy
                ? 'bg-emerald-100 text-emerald-700'
                : 'bg-red-100 text-red-700'
            }`}
          >
            {action.action}
          </span>
          <span className="font-semibold text-gray-900">{action.symbol}</span>
          <span className="text-gray-500">{action.quantity} shares</span>
        </div>
      </div>

      {action.reasoning && (
        <p className="mt-1.5 text-xs text-gray-600">{action.reasoning}</p>
      )}

      {!action.executed && (
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

      {action.executed && (
        <div className="mt-2 rounded-lg bg-gray-200 py-1.5 text-center text-xs font-medium text-gray-600">
          Order Executed
        </div>
      )}
    </div>
  );
}
