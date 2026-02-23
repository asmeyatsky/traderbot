import { useState } from 'react';
import {
  BoltIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  XCircleIcon,
  ShieldExclamationIcon,
  ArrowTrendingDownIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';
import { useTradingActivity } from '../hooks/use-auto-trading';
import { formatCurrency, formatDateTime } from '../lib/format';
import LoadingSpinner from '../components/common/LoadingSpinner';
import PageHeader from '../components/common/PageHeader';
import { PAGE_DESCRIPTIONS } from '../lib/help-text';

const FILTERS = [
  { label: 'All', value: '' },
  { label: 'Signals', value: 'SIGNAL_GENERATED' },
  { label: 'Orders', value: 'ORDER_PLACED' },
  { label: 'Filled', value: 'ORDER_FILLED' },
  { label: 'Failed', value: 'ORDER_FAILED' },
  { label: 'Risk Blocked', value: 'RISK_BLOCKED' },
  { label: 'Stop-Loss', value: 'STOP_LOSS_TRIGGERED' },
  { label: 'Take-Profit', value: 'TAKE_PROFIT_TRIGGERED' },
];

const EVENT_ICONS: Record<string, typeof BoltIcon> = {
  SIGNAL_GENERATED: BoltIcon,
  ORDER_PLACED: DocumentTextIcon,
  ORDER_FILLED: CheckCircleIcon,
  ORDER_FAILED: XCircleIcon,
  RISK_BLOCKED: ShieldExclamationIcon,
  STOP_LOSS_TRIGGERED: ArrowTrendingDownIcon,
  TAKE_PROFIT_TRIGGERED: ArrowTrendingUpIcon,
};

const EVENT_COLORS: Record<string, string> = {
  SIGNAL_GENERATED: 'bg-blue-100 text-blue-700',
  ORDER_PLACED: 'bg-indigo-100 text-indigo-700',
  ORDER_FILLED: 'bg-green-100 text-green-700',
  ORDER_FAILED: 'bg-red-100 text-red-700',
  RISK_BLOCKED: 'bg-amber-100 text-amber-700',
  STOP_LOSS_TRIGGERED: 'bg-red-100 text-red-700',
  TAKE_PROFIT_TRIGGERED: 'bg-emerald-100 text-emerald-700',
};

const PAGE_SIZE = 20;

export default function TradingActivityPage() {
  const [filter, setFilter] = useState('');
  const [skip, setSkip] = useState(0);
  const { data, isLoading } = useTradingActivity(skip, PAGE_SIZE, filter || undefined);

  const hasMore = data ? skip + PAGE_SIZE < data.count : false;

  return (
    <div className="space-y-6">
      <PageHeader title="Trading Activity" description={PAGE_DESCRIPTIONS.activity} />

      <div className="flex flex-wrap gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.value}
            onClick={() => { setFilter(f.value); setSkip(0); }}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              filter === f.value
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <LoadingSpinner className="py-20" />
      ) : !data?.items.length ? (
        <div className="rounded-lg bg-white p-12 text-center shadow-sm ring-1 ring-gray-900/5">
          <BoltIcon className="mx-auto h-10 w-10 text-gray-300" />
          <p className="mt-3 text-sm text-gray-500">No trading activity yet</p>
          <p className="mt-1 text-xs text-gray-400">
            Enable auto-trading in Settings to start generating activity.
          </p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-gray-900/5">
          <ul className="divide-y divide-gray-100">
            {data.items.map((item) => {
              const Icon = EVENT_ICONS[item.event_type] ?? BoltIcon;
              const badgeColor = EVENT_COLORS[item.event_type] ?? 'bg-gray-100 text-gray-700';
              return (
                <li key={item.id} className="flex items-start gap-4 px-6 py-4">
                  <Icon className="mt-0.5 h-5 w-5 shrink-0 text-gray-400" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${badgeColor}`}>
                        {item.event_type.replace(/_/g, ' ')}
                      </span>
                      {item.symbol && (
                        <span className="text-sm font-semibold text-gray-900">{item.symbol}</span>
                      )}
                      {item.signal && (
                        <span className="text-xs text-gray-500">{item.signal}</span>
                      )}
                    </div>
                    <p className="mt-1 text-sm text-gray-600">{item.message}</p>
                    <div className="mt-1 flex flex-wrap gap-4 text-xs text-gray-400">
                      <span>{formatDateTime(item.created_at)}</span>
                      {item.confidence != null && (
                        <span>
                          Confidence:{' '}
                          <span className="font-medium text-gray-600">
                            {(item.confidence * 100).toFixed(0)}%
                          </span>
                        </span>
                      )}
                      {item.quantity != null && (
                        <span>Qty: <span className="font-medium text-gray-600">{item.quantity}</span></span>
                      )}
                      {item.price != null && (
                        <span>Price: <span className="font-medium text-gray-600">{formatCurrency(item.price)}</span></span>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>

          {(skip > 0 || hasMore) && (
            <div className="flex items-center justify-between border-t border-gray-100 px-6 py-3">
              <button
                onClick={() => setSkip((s) => Math.max(0, s - PAGE_SIZE))}
                disabled={skip === 0}
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500 disabled:text-gray-300"
              >
                Previous
              </button>
              <span className="text-xs text-gray-400">
                {skip + 1}–{Math.min(skip + PAGE_SIZE, data.count)} of {data.count}
              </span>
              <button
                onClick={() => setSkip((s) => s + PAGE_SIZE)}
                disabled={!hasMore}
                className="text-sm font-medium text-indigo-600 hover:text-indigo-500 disabled:text-gray-300"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
