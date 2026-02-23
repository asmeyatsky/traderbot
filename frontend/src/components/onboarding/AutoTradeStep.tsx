import { useState } from 'react';
import { MagnifyingGlassIcon, CpuChipIcon, BoltIcon } from '@heroicons/react/24/outline';
import { useUpdateAutoTrading } from '../../hooks/use-auto-trading';
import InfoTooltip from '../common/InfoTooltip';
import { RISK_PRESETS, type RiskPresetKey } from '../../lib/constants';
import { AUTO_TRADING_CONTROL_HELP, HOW_IT_WORKS_STEPS } from '../../lib/help-text';

const STEP_ICONS = [MagnifyingGlassIcon, CpuChipIcon, BoltIcon];

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
  const [takeProfit, setTakeProfit] = useState(15);
  const [confidence, setConfidence] = useState(65);
  const [maxPosition, setMaxPosition] = useState(20);
  const [riskPreset, setRiskPreset] = useState<RiskPresetKey>('MODERATE');
  const { mutate, isPending } = useUpdateAutoTrading();

  function applyPreset(key: Exclude<RiskPresetKey, 'CUSTOM'>) {
    const p = RISK_PRESETS[key];
    setStopLoss(p.stopLoss);
    setTakeProfit(p.takeProfit);
    setConfidence(p.confidence);
    setMaxPosition(p.maxPosition);
    setRiskPreset(key);
  }

  function handleSliderChange(setter: (v: number) => void) {
    return (e: React.ChangeEvent<HTMLInputElement>) => {
      setter(Number(e.target.value));
      setRiskPreset('CUSTOM');
    };
  }

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

  const budgetNum = budget ? Number(budget) : 0;
  const positionValue = budgetNum > 0 ? budgetNum * (maxPosition / 100) : 0;

  return (
    <div className="animate-fade-in text-center">
      <h2 className="text-2xl font-bold text-gray-900">Activate Your AI Trader</h2>
      <p className="mt-2 text-sm text-gray-600">
        TraderBot finds opportunities, executes trades, and manages risk — all within your rules.
      </p>

      {/* How It Works */}
      <div className="mx-auto mt-6 grid max-w-md grid-cols-3 gap-2">
        {HOW_IT_WORKS_STEPS.map((step, i) => {
          const Icon = STEP_ICONS[i];
          return (
            <div key={step.title} className="relative flex flex-col items-center text-center">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-indigo-100">
                <Icon className="h-5 w-5 text-indigo-600" />
              </div>
              <p className="mt-1.5 text-xs font-semibold text-gray-900">{step.title}</p>
              <p className="mt-0.5 text-[11px] leading-tight text-gray-500">{step.description}</p>
              {i < HOW_IT_WORKS_STEPS.length - 1 && (
                <span className="absolute right-0 top-3 translate-x-1/2 text-sm text-gray-300">&rarr;</span>
              )}
            </div>
          );
        })}
      </div>

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
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-800">Risk Controls</h3>
            {riskPreset === 'CUSTOM' && <span className="text-xs font-medium text-amber-600">Custom</span>}
          </div>

          {/* Risk Presets */}
          <div className="flex gap-2">
            {(['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => applyPreset(key)}
                className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  riskPreset === key
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {key.charAt(0) + key.slice(1).toLowerCase()}
              </button>
            ))}
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
                Stop-Loss <InfoTooltip text={AUTO_TRADING_CONTROL_HELP.stop_loss} />
              </label>
              <span className="text-sm font-semibold text-gray-900">{stopLoss}%</span>
            </div>
            <input
              type="range"
              min={1}
              max={25}
              value={stopLoss}
              onChange={handleSliderChange(setStopLoss)}
              className="mt-1 w-full accent-indigo-600"
            />
            {positionValue > 0 && (
              <p className="text-xs text-indigo-600">
                On a ${positionValue.toLocaleString()} position, auto-sell below $
                {Math.round(positionValue * (1 - stopLoss / 100)).toLocaleString()}
              </p>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
                Take-Profit <InfoTooltip text={AUTO_TRADING_CONTROL_HELP.take_profit} />
              </label>
              <span className="text-sm font-semibold text-gray-900">{takeProfit}%</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              value={takeProfit}
              onChange={handleSliderChange(setTakeProfit)}
              className="mt-1 w-full accent-indigo-600"
            />
            {positionValue > 0 && (
              <p className="text-xs text-indigo-600">
                On a ${positionValue.toLocaleString()} position, auto-sell above $
                {Math.round(positionValue * (1 + takeProfit / 100)).toLocaleString()}
              </p>
            )}
          </div>

          <div>
            <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
              Budget <InfoTooltip text={AUTO_TRADING_CONTROL_HELP.budget} />
              <span className="text-gray-400">(optional)</span>
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
              <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
                Confidence Threshold <InfoTooltip text={AUTO_TRADING_CONTROL_HELP.confidence} />
              </label>
              <span className="text-sm font-semibold text-gray-900">{confidence}%</span>
            </div>
            <input
              type="range"
              min={50}
              max={95}
              value={confidence}
              onChange={handleSliderChange(setConfidence)}
              className="mt-1 w-full accent-indigo-600"
            />
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-1 text-sm font-medium text-gray-700">
                Max Position Size <InfoTooltip text={AUTO_TRADING_CONTROL_HELP.max_position} />
              </label>
              <span className="text-sm font-semibold text-gray-900">{maxPosition}%</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              value={maxPosition}
              onChange={handleSliderChange(setMaxPosition)}
              className="mt-1 w-full accent-indigo-600"
            />
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
