import { useState } from 'react';
import { useMarkets, useStocks } from '../hooks/use-markets';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import ScreenerPanel from '../components/markets/ScreenerPanel';
import type { Market } from '../types/market';

export default function MarketsPage() {
  const [selectedMarket, setSelectedMarket] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const { data: marketsData, isLoading, error } = useMarkets();
  const { data: stocksData } = useStocks(selectedMarket, search);

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorAlert message="Failed to load markets" />;

  const markets: Market[] = marketsData?.markets ?? [];
  const stocks = stocksData?.stocks ?? [];

  return (
    <div className="h-full overflow-auto p-4 sm:p-6">
      <h1 className="text-lg font-semibold text-gray-800 dark:text-white">Markets</h1>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Browse markets and discover trading opportunities.
      </p>

      {/* Market selector */}
      <div className="mt-4 flex flex-wrap gap-2">
        {markets.map((market) => (
          <button
            key={market.market_code}
            onClick={() => setSelectedMarket(market.market_code)}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
              selectedMarket === market.market_code
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {market.market_name}
          </button>
        ))}
      </div>

      {/* Search */}
      {selectedMarket && (
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search stocks..."
          className="mt-4 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:max-w-xs dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder-gray-400"
        />
      )}

      {/* Stock list */}
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {stocks.map((stock) => (
          <div
            key={stock.symbol}
            className="rounded-xl border border-gray-200 bg-white p-4 transition-shadow hover:shadow-md dark:border-gray-700 dark:bg-gray-800"
          >
            <p className="font-semibold text-gray-900 dark:text-white">{stock.symbol}</p>
            <p className="text-sm text-gray-500 dark:text-gray-400">{stock.name}</p>
            <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">{stock.sector}</p>
          </div>
        ))}

        {selectedMarket && stocks.length === 0 && (
          <p className="col-span-full py-8 text-center text-sm text-gray-500 dark:text-gray-400">
            No stocks found. Try a different search.
          </p>
        )}

        {!selectedMarket && (
          <p className="col-span-full py-12 text-center text-sm text-gray-500 dark:text-gray-400">
            Select a market to browse stocks.
          </p>
        )}
      </div>

      {/* Stock Screener */}
      <ScreenerPanel />
    </div>
  );
}
