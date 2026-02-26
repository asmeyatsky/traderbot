import { useState, useEffect, useRef, useMemo } from 'react';
import { useCreateOrder } from '../../hooks/use-orders';
import { useMarkets, useStocks } from '../../hooks/use-markets';
import { useEnhancedMarketData } from '../../hooks/use-market-data';
import { ORDER_SIDES, ORDER_TYPES } from '../../lib/constants';
import { ORDER_TYPE_HELP, ORDER_SIDE_HELP } from '../../lib/help-text';
import { formatCurrency, formatPercent, formatNumber } from '../../lib/format';
import InfoTooltip from '../common/InfoTooltip';
import ConfirmDialog from '../common/ConfirmDialog';
import type { Stock } from '../../types/market';
import type { NewsArticle } from '../../types/market-data';

export default function OrderEntryForm() {
  const [selectedMarket, setSelectedMarket] = useState<string>('');
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [stockSearch, setStockSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [showStockDropdown, setShowStockDropdown] = useState(false);
  const [side, setSide] = useState<string>('BUY');
  const [orderType, setOrderType] = useState<string>('MARKET');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [showConfirm, setShowConfirm] = useState(false);
  const { mutate, isPending, error } = useCreateOrder();
  const { data: marketsData } = useMarkets();
  const { data: stocksData, isLoading: stocksLoading } = useStocks(selectedMarket || null, debouncedSearch);

  const { data: marketData, isLoading: marketDataLoading } = useEnhancedMarketData(selectedStock?.symbol ?? '');

  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(stockSearch), 300);
    return () => clearTimeout(timer);
  }, [stockSearch]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowStockDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Reset stock when market changes
  useEffect(() => {
    setSelectedStock(null);
    setStockSearch('');
    setDebouncedSearch('');
  }, [selectedMarket]);

  const markets = marketsData?.markets ?? [];
  const stocks = useMemo(() => stocksData?.stocks ?? [], [stocksData]);

  function handleMarketChange(code: string) {
    setSelectedMarket(code);
  }

  function handleSelectStock(stock: Stock) {
    setSelectedStock(stock);
    setStockSearch('');
    setShowStockDropdown(false);
  }

  function handleClearStock() {
    setSelectedStock(null);
    setStockSearch('');
    setTimeout(() => searchInputRef.current?.focus(), 0);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setShowConfirm(true);
  }

  function handleConfirm() {
    if (!selectedStock) return;
    mutate(
      {
        symbol: selectedStock.symbol.toUpperCase(),
        position_type: side === 'SELL' ? 'SHORT' : 'LONG',
        order_type: orderType,
        quantity: Number(quantity),
        ...(orderType === 'LIMIT' && price ? { limit_price: Number(price) } : {}),
        ...((['STOP_LOSS', 'TRAILING_STOP'].includes(orderType)) && price ? { stop_price: Number(price) } : {}),
      },
      {
        onSuccess: () => {
          setSelectedStock(null);
          setStockSearch('');
          setQuantity('');
          setPrice('');
          setShowConfirm(false);
        },
        onError: () => setShowConfirm(false),
      },
    );
  }

  const displaySymbol = selectedStock?.symbol.toUpperCase() ?? '';

  return (
    <>
      <form onSubmit={handleSubmit} className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
        <h3 className="text-lg font-medium text-gray-900">New Order</h3>
        {error && (
          <p className="mt-2 text-sm text-red-600">
            {(error as any).response?.data?.detail ?? (error as Error).message}
          </p>
        )}
        <div className="mt-4 grid grid-cols-2 gap-4">
          {/* Market selector */}
          <div className="col-span-2">
            <label className="block text-sm font-medium text-gray-700">Market</label>
            <select
              value={selectedMarket}
              onChange={(e) => handleMarketChange(e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              <option value="">Select a market...</option>
              {markets.map((m) => (
                <option key={m.market_code} value={m.market_code}>
                  {m.market_name} ({m.currency})
                </option>
              ))}
            </select>
          </div>

          {/* Stock combobox */}
          {selectedMarket && (
            <div className="col-span-2" ref={dropdownRef}>
              <label className="block text-sm font-medium text-gray-700">Stock</label>
              {selectedStock ? (
                <div className="mt-1 flex items-center gap-2">
                  <span className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-3 py-1.5 text-sm font-medium text-indigo-700 ring-1 ring-indigo-200">
                    <span className="font-semibold">{selectedStock.symbol}</span>
                    <span className="text-indigo-500">— {selectedStock.name}</span>
                    <button
                      type="button"
                      onClick={handleClearStock}
                      className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full text-indigo-400 hover:bg-indigo-200 hover:text-indigo-600"
                    >
                      ×
                    </button>
                  </span>
                </div>
              ) : (
                <div className="relative mt-1">
                  <input
                    ref={searchInputRef}
                    type="text"
                    value={stockSearch}
                    onChange={(e) => {
                      setStockSearch(e.target.value);
                      setShowStockDropdown(true);
                    }}
                    onFocus={() => setShowStockDropdown(true)}
                    placeholder="Search by symbol or name..."
                    required
                    className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
                  />
                  {showStockDropdown && (
                    <ul className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-sm shadow-lg ring-1 ring-black/5">
                      {stocksLoading ? (
                        <li className="px-3 py-2 text-gray-400">Loading...</li>
                      ) : stocks.length === 0 ? (
                        <li className="px-3 py-2 text-gray-400">No matches found</li>
                      ) : (
                        stocks.map((s) => (
                          <li
                            key={s.symbol}
                            onClick={() => handleSelectStock(s)}
                            className="cursor-pointer px-3 py-2 hover:bg-indigo-50"
                          >
                            <span className="font-medium text-gray-900">{s.symbol}</span>
                            <span className="ml-2 text-gray-500">— {s.name}</span>
                            <span className="ml-2 text-xs text-gray-400">({s.sector})</span>
                          </li>
                        ))
                      )}
                    </ul>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Price card */}
          {selectedStock && (
            <div className="col-span-2 rounded-md border border-gray-200 bg-gray-50 p-3">
              {marketDataLoading ? (
                <div className="animate-pulse space-y-2">
                  <div className="h-6 w-28 rounded bg-gray-200" />
                  <div className="flex gap-4">
                    <div className="h-4 w-32 rounded bg-gray-200" />
                    <div className="h-4 w-24 rounded bg-gray-200" />
                  </div>
                </div>
              ) : marketData ? (
                <div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-xl font-semibold text-gray-900">
                      {formatCurrency(marketData.current_price)}
                    </span>
                    <span className={`text-sm font-medium ${marketData.price_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {marketData.price_change >= 0 ? '+' : ''}{formatNumber(marketData.price_change, 2)}
                      {' '}({formatPercent(marketData.price_change_percent)})
                    </span>
                  </div>
                  <div className="mt-1 flex gap-4 text-xs text-gray-500">
                    <span>Day Range: {formatCurrency(marketData.low)} — {formatCurrency(marketData.high)}</span>
                    <span>Vol: {marketData.volume.toLocaleString()}</span>
                  </div>
                </div>
              ) : null}
            </div>
          )}

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
          disabled={isPending || !selectedStock}
          className={`mt-4 w-full rounded-md px-4 py-2 text-sm font-medium text-white shadow-sm ${
            side === 'BUY'
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-red-600 hover:bg-red-700'
          } disabled:opacity-50`}
        >
          {isPending ? 'Placing...' : `${side} ${displaySymbol || 'Order'}`}
        </button>
      </form>

      {selectedStock && marketData?.sentiment && marketData.sentiment.articles.length > 0 && (
        <div className="mt-4 rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-gray-900">Recent News</h4>
            <SentimentBadge label={marketData.sentiment.sentiment_label} />
          </div>
          <ul className="mt-3 divide-y divide-gray-100">
            {marketData.sentiment.articles.slice(0, 5).map((article, i) => (
              <NewsItem key={i} article={article} />
            ))}
          </ul>
        </div>
      )}

      <ConfirmDialog
        open={showConfirm}
        title="Confirm Order"
        confirmLabel={`${side} ${displaySymbol}`}
        confirmColor={side === 'BUY' ? 'green' : 'red'}
        onConfirm={handleConfirm}
        onCancel={() => setShowConfirm(false)}
        isPending={isPending}
      >
        <dl className="space-y-2">
          <div className="flex justify-between">
            <dt className="text-gray-500">Symbol</dt>
            <dd className="font-medium text-gray-900">{displaySymbol}</dd>
          </div>
          {selectedStock && (
            <div className="flex justify-between">
              <dt className="text-gray-500">Company</dt>
              <dd className="font-medium text-gray-900">{selectedStock.name}</dd>
            </div>
          )}
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

function SentimentBadge({ label }: { label: string }) {
  const color =
    label.toLowerCase() === 'bullish'
      ? 'bg-green-50 text-green-700 ring-green-600/20'
      : label.toLowerCase() === 'bearish'
        ? 'bg-red-50 text-red-700 ring-red-600/20'
        : 'bg-gray-50 text-gray-600 ring-gray-500/10';
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${color}`}>
      {label}
    </span>
  );
}

function NewsItem({ article }: { article: NewsArticle }) {
  const sentimentDot =
    article.sentiment_score > 0.2
      ? 'bg-green-400'
      : article.sentiment_score < -0.2
        ? 'bg-red-400'
        : 'bg-gray-300';

  const timeAgo = getTimeAgo(article.published_at);

  return (
    <li className="py-2">
      <div className="flex items-start gap-2">
        <span className={`mt-1.5 h-2 w-2 flex-shrink-0 rounded-full ${sentimentDot}`} />
        <div className="min-w-0">
          <a
            href={article.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-gray-900 hover:text-indigo-600"
          >
            {article.title}
          </a>
          <div className="mt-0.5 text-xs text-gray-500">
            {article.source} &middot; {timeAgo}
          </div>
        </div>
      </div>
    </li>
  );
}

function getTimeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMin = Math.round((now - then) / 60_000);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDays = Math.round(diffHr / 24);
  return `${diffDays}d ago`;
}
