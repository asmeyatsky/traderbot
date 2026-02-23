import { Link } from 'react-router-dom';
import { Cog6ToothIcon } from '@heroicons/react/24/outline';
import { useAutoTradingSettings, useTradingActivitySummary } from '../../hooks/use-auto-trading';
import { formatCurrency } from '../../lib/format';
import { AUTO_TRADING_HELP } from '../../lib/help-text';
import InfoTooltip from '../common/InfoTooltip';

export default function AutoTradingStatusCard() {
  const { data: settings } = useAutoTradingSettings();
  const { data: summary } = useTradingActivitySummary();

  if (!settings) return null;

  const counts = summary?.summary ?? {};

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-1 text-sm font-semibold text-gray-900">
          Auto-Trading
          <InfoTooltip text={AUTO_TRADING_HELP} />
        </h2>
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${
              settings.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            {settings.enabled ? 'Active' : 'Inactive'}
          </span>
          <Link to="/settings" className="text-gray-400 hover:text-gray-600">
            <Cog6ToothIcon className="h-5 w-5" />
          </Link>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {settings.watchlist.length > 0 ? (
          settings.watchlist.map((sym) => (
            <span
              key={sym}
              className="inline-flex rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700"
            >
              {sym}
            </span>
          ))
        ) : (
          <span className="text-xs text-gray-400">
            No watchlist configured. <Link to="/settings" className="text-indigo-600 hover:text-indigo-500">Add stocks in Settings</Link> to start.
          </span>
        )}
      </div>

      {/* Settings at a glance */}
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500">
        {settings.trading_budget != null && (
          <span>Budget: <span className="font-medium text-gray-700">{formatCurrency(settings.trading_budget)}</span></span>
        )}
        <span>SL: <span className="font-medium text-gray-700">{settings.stop_loss_pct}%</span></span>
        <span>TP: <span className="font-medium text-gray-700">{settings.take_profit_pct}%</span></span>
        <span>Confidence: <span className="font-medium text-gray-700">{Math.round(settings.confidence_threshold * 100)}%</span></span>
      </div>

      <div className="mt-4 grid grid-cols-5 gap-2 border-t border-gray-100 pt-4">
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-900">{counts['SIGNAL_GENERATED'] ?? 0}</p>
          <p className="text-xs text-gray-500">Signals</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-900">{counts['ORDER_PLACED'] ?? 0}</p>
          <p className="text-xs text-gray-500">Orders</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-900">{counts['ORDER_FILLED'] ?? 0}</p>
          <p className="text-xs text-gray-500">Filled</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-red-600">{counts['STOP_LOSS_TRIGGERED'] ?? 0}</p>
          <p className="text-xs text-gray-500">Stop-Loss</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-green-600">{counts['TAKE_PROFIT_TRIGGERED'] ?? 0}</p>
          <p className="text-xs text-gray-500">Take-Profit</p>
        </div>
      </div>
    </div>
  );
}
