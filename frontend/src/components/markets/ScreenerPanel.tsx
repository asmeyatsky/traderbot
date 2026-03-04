import { useState } from 'react';
import { usePrebuiltScreens, useScreenStocks } from '../../hooks/use-screening';
import type { ScreenResult } from '../../types/screening';

const SCREEN_COLORS: Record<string, string> = {
  top_gainers: 'bg-green-100 text-green-700 hover:bg-green-200',
  top_losers: 'bg-red-100 text-red-700 hover:bg-red-200',
  most_active: 'bg-blue-100 text-blue-700 hover:bg-blue-200',
  high_momentum: 'bg-purple-100 text-purple-700 hover:bg-purple-200',
  oversold_rsi: 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200',
};

export default function ScreenerPanel() {
  const { data: prebuiltData } = usePrebuiltScreens();
  const screenMutation = useScreenStocks();
  const [activeScreen, setActiveScreen] = useState<string | null>(null);

  const screens = prebuiltData?.screens ?? [];
  const results: ScreenResult[] = screenMutation.data?.results ?? [];

  function handleScreenClick(name: string) {
    setActiveScreen(name);
    screenMutation.mutate({ prebuilt_screen: name, limit: 15 });
  }

  return (
    <div className="mt-6">
      <h2 className="text-sm font-semibold text-gray-700">Stock Screener</h2>
      <div className="mt-2 flex flex-wrap gap-2">
        {screens.map((s) => (
          <button
            key={s.name}
            onClick={() => handleScreenClick(s.name)}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${
              activeScreen === s.name
                ? SCREEN_COLORS[s.name] ?? 'bg-indigo-100 text-indigo-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      {screenMutation.isPending && (
        <p className="mt-4 text-sm text-gray-500">Loading...</p>
      )}

      {results.length > 0 && (
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-xs text-gray-500">
                <th className="pb-2 pr-4 font-medium">Symbol</th>
                <th className="pb-2 pr-4 font-medium">Name</th>
                <th className="pb-2 pr-4 font-medium text-right">Price</th>
                <th className="pb-2 pr-4 font-medium text-right">Change</th>
                <th className="pb-2 font-medium text-right">Volume</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.symbol} className="border-b border-gray-100">
                  <td className="py-2 pr-4 font-semibold text-gray-900">{r.symbol}</td>
                  <td className="py-2 pr-4 text-gray-600">{r.name}</td>
                  <td className="py-2 pr-4 text-right text-gray-900">${r.price.toFixed(2)}</td>
                  <td
                    className={`py-2 pr-4 text-right font-medium ${
                      r.change_pct >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {r.change_pct >= 0 ? '+' : ''}{r.change_pct.toFixed(2)}%
                  </td>
                  <td className="py-2 text-right text-gray-600">
                    {r.volume.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
