import { useState, useEffect } from 'react';
import { useCreateOrder } from '../../hooks/use-orders';

interface TradeDialogProps {
  open: boolean;
  onClose: () => void;
  initialSymbol?: string;
}

export default function TradeDialog({ open, onClose, initialSymbol = '' }: TradeDialogProps) {
  const [symbol, setSymbol] = useState(initialSymbol);
  const [side, setSide] = useState<'BUY' | 'SELL'>('BUY');
  const [quantity, setQuantity] = useState('');
  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('MARKET');
  const [limitPrice, setLimitPrice] = useState('');
  const [error, setError] = useState('');

  const { mutate: createOrder, isPending } = useCreateOrder();

  useEffect(() => {
    if (open) {
      setSymbol(initialSymbol);
      setSide('BUY');
      setQuantity('');
      setOrderType('MARKET');
      setLimitPrice('');
      setError('');
    }
  }, [open, initialSymbol]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    const sym = symbol.trim().toUpperCase();
    if (!sym) { setError('Symbol is required'); return; }
    const qty = parseInt(quantity, 10);
    if (!qty || qty <= 0) { setError('Quantity must be a positive number'); return; }
    if (orderType === 'LIMIT' && (!limitPrice || parseFloat(limitPrice) <= 0)) {
      setError('Limit price must be a positive number');
      return;
    }

    createOrder(
      {
        symbol: sym,
        position_type: side === 'BUY' ? 'LONG' : 'SHORT',
        order_type: orderType,
        quantity: qty,
        ...(orderType === 'LIMIT' ? { limit_price: parseFloat(limitPrice) } : {}),
      },
      {
        onSuccess: () => onClose(),
        onError: (err) => setError(err instanceof Error ? err.message : 'Order failed'),
      },
    );
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="fixed inset-0 bg-black/40" onClick={onClose} />
      <div className="relative w-full max-w-md rounded-lg bg-white dark:bg-gray-800 p-6 shadow-xl">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">New Trade</h3>

        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          {/* Symbol */}
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">Symbol</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
            />
          </div>

          {/* Side toggle */}
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">Side</label>
            <div className="mt-1 flex gap-2">
              <button
                type="button"
                onClick={() => setSide('BUY')}
                className={`flex-1 rounded-md py-2 text-sm font-medium transition-colors ${
                  side === 'BUY'
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                Buy
              </button>
              <button
                type="button"
                onClick={() => setSide('SELL')}
                className={`flex-1 rounded-md py-2 text-sm font-medium transition-colors ${
                  side === 'SELL'
                    ? 'bg-red-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                Sell
              </button>
            </div>
          </div>

          {/* Quantity */}
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">Quantity</label>
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              min="1"
              placeholder="10"
              className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
            />
          </div>

          {/* Order Type */}
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">Order Type</label>
            <div className="mt-1 flex gap-2">
              <button
                type="button"
                onClick={() => setOrderType('MARKET')}
                className={`flex-1 rounded-md py-2 text-sm font-medium transition-colors ${
                  orderType === 'MARKET'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                Market
              </button>
              <button
                type="button"
                onClick={() => setOrderType('LIMIT')}
                className={`flex-1 rounded-md py-2 text-sm font-medium transition-colors ${
                  orderType === 'LIMIT'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                Limit
              </button>
            </div>
          </div>

          {/* Limit Price (conditional) */}
          {orderType === 'LIMIT' && (
            <div>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400">Limit Price</label>
              <input
                type="number"
                value={limitPrice}
                onChange={(e) => setLimitPrice(e.target.value)}
                min="0.01"
                step="0.01"
                placeholder="150.00"
                className="mt-1 w-full rounded-md border border-gray-300 dark:border-gray-600 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 focus:outline-none dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
              />
            </div>
          )}

          {/* Error */}
          {error && (
            <p className="text-xs text-red-600">{error}</p>
          )}

          {/* Buttons */}
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isPending}
              className="rounded-md bg-white dark:bg-gray-800 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 shadow-sm ring-1 ring-gray-300 dark:ring-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isPending}
              className={`rounded-md px-4 py-2 text-sm font-medium text-white shadow-sm disabled:opacity-50 ${
                side === 'BUY'
                  ? 'bg-emerald-600 hover:bg-emerald-700'
                  : 'bg-red-600 hover:bg-red-700'
              }`}
            >
              {isPending ? 'Placing...' : `${side === 'BUY' ? 'Buy' : 'Sell'} ${symbol || 'Stock'}`}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
