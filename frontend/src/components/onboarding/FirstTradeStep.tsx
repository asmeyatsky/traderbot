import { useState, useEffect, useRef, useMemo } from 'react';
import { useCreateOrder } from '../../hooks/use-orders';
import { useMarkets, useStocks } from '../../hooks/use-markets';
import { useEnhancedMarketData } from '../../hooks/use-market-data';
import { formatCurrency, formatPercent } from '../../lib/format';
import type { Stock } from '../../types/market';

interface FirstTradeStepProps {
  symbol: string;
  onNext: () => void;
}

export default function FirstTradeStep({ symbol: initialSymbol, onNext }: FirstTradeStepProps) {
  const [selectedMarket, setSelectedMarket] = useState('');
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [stockSearch, setStockSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [quantity, setQuantity] = useState(1);
  const [success, setSuccess] = useState(false);
  const { mutate, isPending, error } = useCreateOrder();
  const { data: marketsData } = useMarkets();
  const { data: stocksData, isLoading: stocksLoading } = useStocks(selectedMarket || null, debouncedSearch);

  const { data: marketData } = useEnhancedMarketData(selectedStock?.symbol ?? '');

  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(stockSearch), 300);
    return () => clearTimeout(timer);
  }, [stockSearch]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    setSelectedStock(null);
    setStockSearch('');
    setDebouncedSearch('');
  }, [selectedMarket]);

  // Auto-select first market
  useEffect(() => {
    if (marketsData?.markets?.length && !selectedMarket) {
      setSelectedMarket(marketsData.markets[0].market_code);
    }
  }, [marketsData, selectedMarket]);

  const markets = marketsData?.markets ?? [];
  const stocks = useMemo(() => stocksData?.stocks ?? [], [stocksData]);

  // Auto-select the stock passed from ExploreStep if it exists in the loaded stocks
  useEffect(() => {
    if (initialSymbol && stocks.length > 0 && !selectedStock) {
      const match = stocks.find((s) => s.symbol.toUpperCase() === initialSymbol.toUpperCase());
      if (match) setSelectedStock(match);
    }
  }, [initialSymbol, stocks, selectedStock]);

  const tradeSymbol = selectedStock?.symbol ?? initialSymbol;

  function handleSelectStock(stock: Stock) {
    setSelectedStock(stock);
    setStockSearch('');
    setShowDropdown(false);
  }

  function handleClearStock() {
    setSelectedStock(null);
    setStockSearch('');
  }

  function handleTrade() {
    mutate(
      { symbol: tradeSymbol.toUpperCase(), position_type: 'LONG', order_type: 'MARKET', quantity },
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
          Your market buy order for {quantity} share{quantity > 1 ? 's' : ''} of {tradeSymbol} has been submitted.
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
          {/* Market selector */}
          <div>
            <div className="text-xs font-medium text-gray-500">Market</div>
            <select
              value={selectedMarket}
              onChange={(e) => setSelectedMarket(e.target.value)}
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
            <div ref={dropdownRef}>
              <div className="text-xs font-medium text-gray-500">Stock</div>
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
                    type="text"
                    value={stockSearch}
                    onChange={(e) => {
                      setStockSearch(e.target.value);
                      setShowDropdown(true);
                    }}
                    onFocus={() => setShowDropdown(true)}
                    placeholder="Search by symbol or name..."
                    className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
                  />
                  {showDropdown && (
                    <ul className="absolute z-10 mt-1 max-h-48 w-full overflow-auto rounded-md bg-white py-1 text-sm shadow-lg ring-1 ring-black/5">
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
                          </li>
                        ))
                      )}
                    </ul>
                  )}
                </div>
              )}
            </div>
          )}

          {selectedStock && marketData && (
            <div className="rounded-md bg-white px-3 py-2 text-sm ring-1 ring-gray-200">
              <span className="text-gray-500">Current Price: </span>
              <span className="font-semibold text-gray-900">{formatCurrency(marketData.current_price)}</span>
              <span className={`ml-1 font-medium ${marketData.price_change_percent >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ({formatPercent(marketData.price_change_percent)})
              </span>
            </div>
          )}

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
          disabled={isPending || !selectedStock}
          className="rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {isPending ? 'Placing order...' : 'Place Order'}
        </button>
      </div>
    </div>
  );
}
