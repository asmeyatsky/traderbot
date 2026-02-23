import { useState } from 'react';
import { useUpdateAutoTrading } from '../../hooks/use-auto-trading';

interface AutoTradeStepProps {
  symbol: string;
  onNext: () => void;
}

export default function AutoTradeStep({ symbol, onNext }: AutoTradeStepProps) {
  const [enabled, setEnabled] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>(symbol ? [symbol] : []);
  const [symbolInput, setSymbolInput] = useState('');
  const [budget, setBudget] = useState('');
  const [stopLoss, setStopLoss] = useState(5);
  const [takeProfit, setTakeProfit] = useState(10);
  const [confidence, setConfidence] = useState(60);
  const [maxPosition, setMaxPosition] = useState(20);
  const { mutate, isPending } = useUpdateAutoTrading();

  function addSymbol() {
    const sym = symbolInput.trim().toUpperCase();
    if (sym && !watchlist.includes(sym)) {
      setWatchlist((w) => [...w, sym]);
    }
    setSymbolInput('');
  }

  function removeSymbol(sym: string) {
    setWatchlist((w) => w.filter((s) => s !== sym));
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      addSymbol();
    }
  }

  function handleSave() {
    mutate(
      {
        enabled,
        watchlist,
        trading_budget: budget ? Number(budget) : null,
        stop_loss_pct: stopLoss,
        take_profit_pct: takeProfit,
        confidence_threshold: confidence / 100,
        max_position_pct: maxPosition,
      },
      { onSuccess: () => onNext() },
    );
  }

  return (
    <div className="animate-fade-in text-center">
      <h2 className="text-2xl font-bold text-gray-900">Put Your Portfolio on Autopilot</h2>
      <p className="mt-2 text-sm text-gray-600">
        Enable autonomous trading and TraderBot will find opportunities, execute trades, and manage risk for you.
      </p>

      <div className="mx-auto mt-8 max-w-sm space-y-6 text-left">
        {/* Section 1: Enable + Watchlist */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700">Enable Auto-Trading</label>
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              onClick={() => setEnabled((v) => !v)}
              className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
                enabled ? 'bg-indigo-600' : 'bg-gray-200'
              }`}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow ring-0 transition-transform ${
                  enabled ? 'translate-x-5' : 'translate-x-0'
                }`}
              />
            </button>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">Watchlist</label>
            <div className="mt-1 flex flex-wrap gap-2">
              {watchlist.map((sym) => (
                <span
                  key={sym}
                  className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
                >
                  {sym}
                  <button type="button" onClick={() => removeSymbol(sym)} className="text-indigo-400 hover:text-indigo-600">
                    &times;
                  </button>
                </span>
              ))}
            </div>
            <div className="mt-2 flex gap-2">
              <input
                value={symbolInput}
                onChange={(e) => setSymbolInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Add symbol"
                className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
              <button
                type="button"
                onClick={addSymbol}
                className="rounded-md bg-gray-100 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200"
              >
                Add
              </button>
            </div>
          </div>
        </div>

        {/* Section 2: Risk Controls */}
        <div className="space-y-4 border-t border-gray-200 pt-4">
          <h3 className="text-sm font-semibold text-gray-800">Risk Controls</h3>

          <div>
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">Stop-Loss</label>
              <span className="text-sm font-semibold text-gray-900">{stopLoss}%</span>
            </div>
            <input
              type="range"
              min={1}
              max={25}
              value={stopLoss}
              onChange={(e) => setStopLoss(Number(e.target.value))}
              className="mt-1 w-full accent-indigo-600"
            />
            <p className="text-xs text-gray-400">Auto-sell if a position drops by this %</p>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">Take-Profit</label>
              <span className="text-sm font-semibold text-gray-900">{takeProfit}%</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              value={takeProfit}
              onChange={(e) => setTakeProfit(Number(e.target.value))}
              className="mt-1 w-full accent-indigo-600"
            />
            <p className="text-xs text-gray-400">Auto-sell if a position gains this %</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Budget <span className="text-gray-400">(optional)</span>
            </label>
            <div className="relative mt-1">
              <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">$</span>
              <input
                type="number"
                min={0}
                step={100}
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                placeholder="e.g. 10000"
                className="block w-full rounded-md border border-gray-300 py-2 pl-7 pr-3 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Section 3: Strategy Tuning */}
        <div className="space-y-4 border-t border-gray-200 pt-4">
          <h3 className="text-sm font-semibold text-gray-800">Strategy Tuning</h3>

          <div>
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">Confidence Threshold</label>
              <span className="text-sm font-semibold text-gray-900">{confidence}%</span>
            </div>
            <input
              type="range"
              min={50}
              max={95}
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
              className="mt-1 w-full accent-indigo-600"
            />
            <p className="text-xs text-gray-400">Higher = fewer but more confident trades</p>
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-gray-700">Max Position Size</label>
              <span className="text-sm font-semibold text-gray-900">{maxPosition}%</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              value={maxPosition}
              onChange={(e) => setMaxPosition(Number(e.target.value))}
              className="mt-1 w-full accent-indigo-600"
            />
            <p className="text-xs text-gray-400">Max % of budget per single stock</p>
          </div>
        </div>
      </div>

      <div className="mt-8 flex items-center justify-center gap-4">
        <button onClick={onNext} className="text-sm font-medium text-gray-500 hover:text-gray-700">
          Skip for Now
        </button>
        <button
          onClick={handleSave}
          disabled={isPending}
          className="rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {isPending ? 'Saving...' : 'Save & Continue'}
        </button>
      </div>
    </div>
  );
}
