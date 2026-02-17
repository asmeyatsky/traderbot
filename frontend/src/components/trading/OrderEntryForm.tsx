import { useState } from 'react';
import { useCreateOrder } from '../../hooks/use-orders';
import { ORDER_SIDES, ORDER_TYPES } from '../../lib/constants';

export default function OrderEntryForm() {
  const [symbol, setSymbol] = useState('');
  const [side, setSide] = useState<string>('BUY');
  const [orderType, setOrderType] = useState<string>('MARKET');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const { mutate, isPending, error } = useCreateOrder();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    mutate(
      {
        symbol: symbol.toUpperCase(),
        side,
        order_type: orderType,
        quantity: Number(quantity),
        ...(orderType !== 'MARKET' && price ? { price: Number(price) } : {}),
      },
      {
        onSuccess: () => {
          setSymbol('');
          setQuantity('');
          setPrice('');
        },
      },
    );
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-lg font-medium text-gray-900">New Order</h3>
      {error && <p className="mt-2 text-sm text-red-600">{(error as Error).message}</p>}
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="col-span-2">
          <label className="block text-sm font-medium text-gray-700">Symbol</label>
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="AAPL"
            required
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Side</label>
          <select
            value={side}
            onChange={(e) => setSide(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
          >
            {ORDER_SIDES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Type</label>
          <select
            value={orderType}
            onChange={(e) => setOrderType(e.target.value)}
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
          >
            {ORDER_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Quantity</label>
          <input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            min={1}
            required
            className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
          />
        </div>
        {orderType !== 'MARKET' && (
          <div>
            <label className="block text-sm font-medium text-gray-700">Price</label>
            <input
              type="number"
              step="0.01"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
        )}
      </div>
      <button
        type="submit"
        disabled={isPending}
        className={`mt-4 w-full rounded-md px-4 py-2 text-sm font-medium text-white shadow-sm ${
          side === 'BUY'
            ? 'bg-green-600 hover:bg-green-700'
            : 'bg-red-600 hover:bg-red-700'
        } disabled:opacity-50`}
      >
        {isPending ? 'Placing...' : `${side} ${symbol.toUpperCase() || 'Order'}`}
      </button>
    </form>
  );
}
