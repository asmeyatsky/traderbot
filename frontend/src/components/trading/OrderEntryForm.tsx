import { useState } from 'react';
import { useCreateOrder } from '../../hooks/use-orders';
import { ORDER_SIDES, ORDER_TYPES } from '../../lib/constants';
import { ORDER_TYPE_HELP, ORDER_SIDE_HELP } from '../../lib/help-text';
import InfoTooltip from '../common/InfoTooltip';
import ConfirmDialog from '../common/ConfirmDialog';

export default function OrderEntryForm() {
  const [symbol, setSymbol] = useState('');
  const [side, setSide] = useState<string>('BUY');
  const [orderType, setOrderType] = useState<string>('MARKET');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [showConfirm, setShowConfirm] = useState(false);
  const { mutate, isPending, error } = useCreateOrder();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setShowConfirm(true);
  }

  function handleConfirm() {
    mutate(
      {
        symbol: symbol.toUpperCase(),
        position_type: side === 'SELL' ? 'SHORT' : 'LONG',
        order_type: orderType,
        quantity: Number(quantity),
        ...(orderType !== 'MARKET' && price ? { limit_price: Number(price) } : {}),
      },
      {
        onSuccess: () => {
          setSymbol('');
          setQuantity('');
          setPrice('');
          setShowConfirm(false);
        },
        onSettled: () => setShowConfirm(false),
      },
    );
  }

  return (
    <>
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
            <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
              Side
              <InfoTooltip text={ORDER_SIDE_HELP[side]} position="bottom" />
            </label>
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
            <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
              Type
              <InfoTooltip text={ORDER_TYPE_HELP[orderType]} position="bottom" />
            </label>
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
            <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
              Quantity
              <InfoTooltip text="Number of shares to buy or sell." position="bottom" />
            </label>
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
              <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
                Price
                <InfoTooltip text="The price at which you want your order to execute." position="bottom" />
              </label>
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

      <ConfirmDialog
        open={showConfirm}
        title="Confirm Order"
        confirmLabel={`${side} ${symbol.toUpperCase()}`}
        confirmColor={side === 'BUY' ? 'green' : 'red'}
        onConfirm={handleConfirm}
        onCancel={() => setShowConfirm(false)}
        isPending={isPending}
      >
        <dl className="space-y-2">
          <div className="flex justify-between">
            <dt className="text-gray-500">Symbol</dt>
            <dd className="font-medium text-gray-900">{symbol.toUpperCase()}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-500">Side</dt>
            <dd className={`font-medium ${side === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>{side}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-500">Type</dt>
            <dd className="font-medium text-gray-900">{orderType}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-gray-500">Quantity</dt>
            <dd className="font-medium text-gray-900">{quantity}</dd>
          </div>
          {orderType !== 'MARKET' && price && (
            <div className="flex justify-between">
              <dt className="text-gray-500">Price</dt>
              <dd className="font-medium text-gray-900">${price}</dd>
            </div>
          )}
        </dl>
        {orderType === 'MARKET' && (
          <p className="mt-3 text-xs text-amber-600">
            Market orders execute at the current price, which may differ from the last quoted price.
          </p>
        )}
      </ConfirmDialog>
    </>
  );
}
