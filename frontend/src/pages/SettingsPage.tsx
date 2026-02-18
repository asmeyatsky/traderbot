import { useState, useEffect } from 'react';
import { useMe, useUpdateMe } from '../hooks/use-auth';
import { useAutoTradingSettings, useUpdateAutoTrading } from '../hooks/use-auto-trading';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { RISK_TOLERANCES, INVESTMENT_GOALS } from '../lib/constants';

export default function SettingsPage() {
  const { data: user, isLoading } = useMe();
  const { mutate: update, isPending, isSuccess } = useUpdateMe();
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

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
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
          </div>
        </div>
        <button
          type="submit"
          disabled={isPending}
          className="mt-6 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {isPending ? 'Saving...' : 'Save Changes'}
        </button>
      </form>

      <AutoTradingSection />
    </div>
  );
}

function AutoTradingSection() {
  const { data: settings, isLoading } = useAutoTradingSettings();
  const { mutate: save, isPending, isSuccess } = useUpdateAutoTrading();
  const [enabled, setEnabled] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [symbolInput, setSymbolInput] = useState('');
  const [budget, setBudget] = useState('');

  useEffect(() => {
    if (settings) {
      setEnabled(settings.enabled);
      setWatchlist(settings.watchlist);
      setBudget(settings.trading_budget != null ? String(settings.trading_budget) : '');
    }
  }, [settings]);

  if (isLoading) return <LoadingSpinner className="py-10" />;

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
    });
  }

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900">Auto-Trading</h2>
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
            enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
          }`}
        >
          {enabled ? 'Active' : 'Inactive'}
        </span>
      </div>

      {isSuccess && (
        <p className="mt-3 rounded-md bg-green-50 p-3 text-sm text-green-700">Auto-trading settings saved</p>
      )}

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

        <div>
          <label className="block text-sm font-medium text-gray-700">
            Trading Budget <span className="text-gray-400">(optional — blank uses full cash balance)</span>
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
