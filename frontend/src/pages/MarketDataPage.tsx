import { useState } from 'react';
import { useEnhancedMarketData } from '../hooks/use-market-data';
import { useTechnicalIndicators } from '../hooks/use-dashboard';
import SymbolSearch from '../components/market-data/SymbolSearch';
import PriceChart from '../components/market-data/PriceChart';
import IndicatorPanel from '../components/market-data/IndicatorPanel';
import StatCard from '../components/common/StatCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { formatCurrency, formatPercent, formatNumber } from '../lib/format';

export default function MarketDataPage() {
  const [symbol, setSymbol] = useState('');
  const { data, isLoading, error } = useEnhancedMarketData(symbol);
  const { data: indicators } = useTechnicalIndicators(symbol);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Market Data</h1>
      <SymbolSearch onSearch={setSymbol} currentSymbol={symbol} />

      {!symbol && (
        <p className="py-10 text-center text-gray-500">Enter a symbol to view market data</p>
      )}

      {isLoading && <LoadingSpinner className="py-10" />}
      {error && <ErrorAlert message={`Failed to load data for ${symbol}`} />}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard title="Price" value={formatCurrency(data.current_price)} />
            <StatCard
              title="Change"
              value={formatCurrency(data.price_change)}
              change={formatPercent(data.price_change_percent)}
              changeType={data.price_change >= 0 ? 'positive' : 'negative'}
            />
            <StatCard title="High" value={formatCurrency(data.high)} />
            <StatCard title="Volume" value={formatNumber(data.volume, 0)} />
          </div>
          {data.historical_data?.length > 0 && (
            <PriceChart data={data.historical_data} symbol={symbol} />
          )}
          {indicators && <IndicatorPanel data={indicators} />}
        </>
      )}
    </div>
  );
}
