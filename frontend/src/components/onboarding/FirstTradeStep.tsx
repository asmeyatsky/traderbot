import { useState } from 'react';
import { useCreateOrder } from '../../hooks/use-orders';

interface FirstTradeStepProps {
  symbol: string;
  onNext: () => void;
}

export default function FirstTradeStep({ symbol, onNext }: FirstTradeStepProps) {
  const [quantity, setQuantity] = useState(1);
  const [success, setSuccess] = useState(false);
  const { mutate, isPending, error } = useCreateOrder();

  function handleTrade() {
    mutate(
      { symbol, position_type: 'LONG', order_type: 'MARKET', quantity },
      { onSuccess: () => setSuccess(true) },
    );
  }

  if (success) {
    return (
      <div className="animate-fade-in text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
          <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
        </div>
        <h3 className="mt-4 text-lg font-semibold text-gray-900">Order Placed!</h3>
        <p className="mt-2 text-sm text-gray-500">
          Your market buy order for {quantity} share{quantity > 1 ? 's' : ''} of {symbol} has been submitted.
        </p>
        <button
          onClick={onNext}
          className="mt-8 rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700"
        >
          Continue
        </button>
      </div>
    );
  }

  return (
    <div className="animate-fade-in text-center">
      <h2 className="text-2xl font-bold text-gray-900">Place Your First Trade</h2>
      <p className="mt-2 text-sm text-gray-600">
        Try placing a simple market buy order. You can always cancel pending orders later.
      </p>

      <div className="mx-auto mt-8 max-w-xs rounded-lg bg-gray-50 p-6 text-left">
        <div className="space-y-4">
          <div>
            <div className="text-xs font-medium text-gray-500">Symbol</div>
            <div className="mt-1 text-lg font-bold text-gray-900">{symbol}</div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs font-medium text-gray-500">Side</div>
              <div className="mt-1 rounded-md bg-green-100 px-3 py-1 text-center text-sm font-semibold text-green-700">
                BUY
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-gray-500">Type</div>
              <div className="mt-1 rounded-md bg-gray-200 px-3 py-1 text-center text-sm font-semibold text-gray-700">
                MARKET
              </div>
            </div>
          </div>
          <div>
            <label htmlFor="qty" className="text-xs font-medium text-gray-500">Quantity</label>
            <input
              id="qty"
              type="number"
              min={1}
              value={quantity}
              onChange={(e) => setQuantity(Math.max(1, Number(e.target.value)))}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {error && (
        <p className="mt-3 text-sm text-red-600">Order failed. Please try again.</p>
      )}

      <div className="mt-8 flex items-center justify-center gap-4">
        <button
          onClick={onNext}
          className="text-sm font-medium text-gray-500 hover:text-gray-700"
        >
          Skip
        </button>
        <button
          onClick={handleTrade}
          disabled={isPending}
          className="rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {isPending ? 'Placing order...' : 'Place Order'}
        </button>
      </div>
    </div>
  );
}
