import { useState, useEffect, useRef, useMemo } from 'react';
import { usePrediction } from '../../hooks/use-ml';
import { useMarkets, useStocks } from '../../hooks/use-markets';
import { formatCurrency } from '../../lib/format';
import LoadingSpinner from '../common/LoadingSpinner';
import type { Stock } from '../../types/market';

interface ExploreStepProps {
  onNext: (symbol: string) => void;
}

export default function ExploreStep({ onNext }: ExploreStepProps) {
  const [selectedMarket, setSelectedMarket] = useState('');
  const [selectedStock, setSelectedStock] = useState<Stock | null>(null);
  const [stockSearch, setStockSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const [searchSymbol, setSearchSymbol] = useState('AAPL');
  const { data, isLoading, error } = usePrediction(searchSymbol);
  const { data: marketsData } = useMarkets();
  const { data: stocksData, isLoading: stocksLoading } = useStocks(selectedMarket || null, debouncedSearch);

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

  function handleSelectStock(stock: Stock) {
    setSelectedStock(stock);
    setStockSearch('');
    setShowDropdown(false);
    setSearchSymbol(stock.symbol.toUpperCase());
  }

  function handleClearStock() {
    setSelectedStock(null);
    setStockSearch('');
  }

  return (
    <div className="animate-fade-in text-center">
      <h2 className="text-2xl font-bold text-gray-900">Explore AI Predictions</h2>
      <p className="mt-2 text-sm text-gray-600">
        See how our AI analyzes stocks. Pick a market and search for a stock.
      </p>

      <div className="mx-auto mt-6 max-w-xs space-y-3 text-left">
        <div>
          <label className="block text-xs font-medium text-gray-500">Market</label>
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

        {selectedMarket && (
          <div ref={dropdownRef}>
            <label className="block text-xs font-medium text-gray-500">Stock</label>
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
      </div>

      <div className="mx-auto mt-6 max-w-sm">
        {isLoading && <LoadingSpinner className="py-8" />}
        {error && (
          <p className="py-4 text-sm text-red-600">Could not load prediction for {searchSymbol}.</p>
        )}
        {data && !isLoading && (
          <div className="rounded-lg bg-gray-50 p-6 text-left">
            <div className="text-center text-lg font-bold text-gray-900">{data.symbol}</div>
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-gray-500">Direction</div>
                <div className={`font-semibold ${data.predicted_direction === 'UP' ? 'text-green-600' : 'text-red-600'}`}>
                  {data.predicted_direction}
                </div>
              </div>
              <div>
                <div className="text-gray-500">Confidence</div>
                <div className="font-semibold text-gray-900">{(data.confidence * 100).toFixed(0)}%</div>
              </div>
              <div>
                <div className="text-gray-500">Current Price</div>
                <div className="font-semibold text-gray-900">{formatCurrency(data.current_price)}</div>
              </div>
              <div>
                <div className="text-gray-500">Predicted Price</div>
                <div className="font-semibold text-gray-900">{formatCurrency(data.predicted_price)}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-lg bg-indigo-50 p-3">
        <p className="text-xs text-indigo-700">
          Predictions are generated by ML models analyzing market data, sentiment, and technical indicators.
          Always do your own research before trading.
        </p>
      </div>

      <button
        onClick={() => onNext(searchSymbol)}
        className="mt-8 rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700"
      >
        Continue
      </button>
    </div>
  );
}
