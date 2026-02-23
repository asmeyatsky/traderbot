import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MagnifyingGlassIcon, CpuChipIcon, BoltIcon } from '@heroicons/react/24/outline';
import { useMe, useUpdateMe } from '../hooks/use-auth';
import { useAutoTradingSettings, useUpdateAutoTrading } from '../hooks/use-auto-trading';
import { usePortfolio } from '../hooks/use-portfolio';
import { useOnboardingStore } from '../stores/onboarding-store';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PageHeader from '../components/common/PageHeader';
import InfoTooltip from '../components/common/InfoTooltip';
import { RISK_TOLERANCES, INVESTMENT_GOALS, RISK_PRESETS, type RiskPresetKey } from '../lib/constants';
import {
  PAGE_DESCRIPTIONS,
  RISK_TOLERANCE_HELP,
  INVESTMENT_GOAL_HELP,
  AUTO_TRADING_HELP,
  AUTO_TRADING_CONTROL_HELP,
  HOW_IT_WORKS_STEPS,
} from '../lib/help-text';

export default function SettingsPage() {
  const navigate = useNavigate();
  const { data: user, isLoading } = useMe();
  const { mutate: update, isPending, isSuccess } = useUpdateMe();
  const resetOnboarding = useOnboardingStore((s) => s.reset);
  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    risk_tolerance: '',
    investment_goal: '',
  });

  useEffect(() => {
    if (user) {
      setForm({
        first_name: user.first_name,
        last_name: user.last_name,
        risk_tolerance: user.risk_tolerance ?? 'MODERATE',
        investment_goal: user.investment_goal ?? 'BALANCED_GROWTH',
      });
    }
  }, [user]);

  if (isLoading) return <LoadingSpinner className="py-20" />;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    update(form);
  }

  function handleRestartOnboarding() {
    resetOnboarding();
    navigate('/onboarding');
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <PageHeader title="Settings" description={PAGE_DESCRIPTIONS.settings} />
      <form onSubmit={handleSubmit} className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
        {isSuccess && (
          <p className="mb-4 rounded-md bg-green-50 p-3 text-sm text-green-700">Settings saved</p>
        )}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">First Name</label>
              <input
                value={form.first_name}
                onChange={(e) => setForm((f) => ({ ...f, first_name: e.target.value }))}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Last Name</label>
              <input
                value={form.last_name}
                onChange={(e) => setForm((f) => ({ ...f, last_name: e.target.value }))}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Email</label>
            <input
              type="email"
              value={user?.email ?? ''}
              disabled
              className="mt-1 block w-full rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-gray-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Risk Tolerance</label>
            <select
              value={form.risk_tolerance}
              onChange={(e) => setForm((f) => ({ ...f, risk_tolerance: e.target.value }))}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              {RISK_TOLERANCES.map((r) => (
                <option key={r} value={r}>{r.replace('_', ' ')}</option>
              ))}
            </select>
            {form.risk_tolerance && RISK_TOLERANCE_HELP[form.risk_tolerance] && (
              <p className="mt-1 text-xs text-gray-500">{RISK_TOLERANCE_HELP[form.risk_tolerance]}</p>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Investment Goal</label>
            <select
              value={form.investment_goal}
              onChange={(e) => setForm((f) => ({ ...f, investment_goal: e.target.value }))}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              {INVESTMENT_GOALS.map((g) => (
                <option key={g} value={g}>{g.replace(/_/g, ' ')}</option>
              ))}
            </select>
            {form.investment_goal && INVESTMENT_GOAL_HELP[form.investment_goal] && (
              <p className="mt-1 text-xs text-gray-500">{INVESTMENT_GOAL_HELP[form.investment_goal]}</p>
            )}
          </div>
        </div>
        <div className="mt-6 flex items-center justify-between">
          <button
            type="submit"
            disabled={isPending}
            className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
          >
            {isPending ? 'Saving...' : 'Save Changes'}
          </button>
          <button
            type="button"
            onClick={handleRestartOnboarding}
            className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
          >
            Restart Onboarding
          </button>
        </div>
      </form>

      <AutoTradingSection />
    </div>
  );
}

const HOW_IT_WORKS_ICONS = [MagnifyingGlassIcon, CpuChipIcon, BoltIcon];

function detectPreset(sl: number, tp: number, conf: number, mp: number): RiskPresetKey {
  for (const [key, vals] of Object.entries(RISK_PRESETS)) {
    if (vals.stopLoss === sl && vals.takeProfit === tp && vals.confidence === conf && vals.maxPosition === mp) {
      return key as RiskPresetKey;
    }
  }
  return 'CUSTOM';
}

function AutoTradingSection() {
  const { data: settings, isLoading } = useAutoTradingSettings();
  const { data: portfolio } = usePortfolio();
  const { mutate: save, isPending, isSuccess } = useUpdateAutoTrading();
  const [enabled, setEnabled] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [symbolInput, setSymbolInput] = useState('');
  const [budget, setBudget] = useState('');
  const [stopLoss, setStopLoss] = useState(5);
  const [takeProfit, setTakeProfit] = useState(15);
  const [confidence, setConfidence] = useState(65);
  const [maxPosition, setMaxPosition] = useState(20);
  const [riskPreset, setRiskPreset] = useState<RiskPresetKey>('MODERATE');

  useEffect(() => {
    if (settings) {
      setEnabled(settings.enabled);
      setWatchlist(settings.watchlist);
      setStopLoss(settings.stop_loss_pct);
      setTakeProfit(settings.take_profit_pct);
      const conf = Math.round(settings.confidence_threshold * 100);
      setConfidence(conf);
      setMaxPosition(settings.max_position_pct);
      setRiskPreset(detectPreset(settings.stop_loss_pct, settings.take_profit_pct, conf, settings.max_position_pct));

      if (settings.trading_budget != null) {
        setBudget(String(settings.trading_budget));
      } else if (portfolio?.cash_balance) {
        setBudget(String(Math.floor(portfolio.cash_balance)));
      }
    }
  }, [settings, portfolio]);

  if (isLoading) return <LoadingSpinner className="py-10" />;

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
    save({
      enabled,
      watchlist,
      trading_budget: budget ? Number(budget) : null,
      stop_loss_pct: stopLoss,
      take_profit_pct: takeProfit,
      confidence_threshold: confidence / 100,
      max_position_pct: maxPosition,
    });
  }

  const budgetNum = budget ? Number(budget) : 0;
  const positionValue = budgetNum > 0 ? budgetNum * (maxPosition / 100) : 0;

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-1 text-lg font-semibold text-gray-900">
          Auto-Trading
          <InfoTooltip text={AUTO_TRADING_HELP} />
        </h2>
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
            enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
          }`}
        >
          {enabled ? 'Active' : 'Inactive'}
        </span>
      </div>
      <p className="mt-1 text-xs text-gray-500">
        Automatically monitor your watchlist and execute trades based on ML signals and your risk settings.
      </p>

      {/* How It Works */}
      <div className="mt-4 grid grid-cols-3 gap-2 rounded-lg bg-gray-50 p-3">
        {HOW_IT_WORKS_STEPS.map((step, i) => {
          const Icon = HOW_IT_WORKS_ICONS[i];
          return (
            <div key={step.title} className="relative flex flex-col items-center text-center">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-indigo-100">
                <Icon className="h-4.5 w-4.5 text-indigo-600" />
              </div>
              <p className="mt-1 text-xs font-semibold text-gray-900">{step.title}</p>
              <p className="mt-0.5 text-[11px] leading-tight text-gray-500">{step.description}</p>
              {i < HOW_IT_WORKS_STEPS.length - 1 && (
                <span className="absolute right-0 top-2.5 translate-x-1/2 text-sm text-gray-300">&rarr;</span>
              )}
            </div>
          );
        })}
      </div>

      {isSuccess && (
        <p className="mt-3 rounded-md bg-green-50 p-3 text-sm text-green-700">Auto-trading settings saved</p>
      )}

      {/* Enable + Watchlist */}
      <div className="mt-4 space-y-4">
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
                className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-3 py-1 text-sm font-medium text-indigo-700"
              >
                {sym}
                <button
                  type="button"
                  onClick={() => removeSymbol(sym)}
                  className="ml-0.5 text-indigo-400 hover:text-indigo-600"
                >
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
              placeholder="Add symbol (e.g. AAPL)"
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

      {/* Risk Controls */}
      <div className="mt-6 space-y-4 border-t border-gray-200 pt-4">
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
            Trading Budget <InfoTooltip text={AUTO_TRADING_CONTROL_HELP.budget} />
            <span className="text-gray-400">(optional — blank uses full cash balance)</span>
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
              className="block w-full rounded-md border border-gray-300 py-2 pl-7 pr-3 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Strategy Tuning */}
      <div className="mt-6 space-y-4 border-t border-gray-200 pt-4">
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

      <button
        type="button"
        onClick={handleSave}
        disabled={isPending}
        className="mt-6 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
      >
        {isPending ? 'Saving...' : 'Save Auto-Trading Settings'}
      </button>
    </div>
  );
}
