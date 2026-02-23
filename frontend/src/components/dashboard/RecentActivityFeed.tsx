import { Link } from 'react-router-dom';
import {
  BoltIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  XCircleIcon,
  ShieldExclamationIcon,
  ArrowTrendingDownIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';
import { useTradingActivity } from '../../hooks/use-auto-trading';

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
  SIGNAL_GENERATED: 'text-blue-500',
  ORDER_PLACED: 'text-indigo-500',
  ORDER_FILLED: 'text-green-500',
  ORDER_FAILED: 'text-red-500',
  RISK_BLOCKED: 'text-amber-500',
  STOP_LOSS_TRIGGERED: 'text-red-500',
  TAKE_PROFIT_TRIGGERED: 'text-emerald-500',
};

function timeAgo(dateStr: string): string {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function RecentActivityFeed() {
  const { data } = useTradingActivity(0, 5);

  if (!data?.items.length) return null;

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-900">Recent Activity</h2>
        <Link to="/activity" className="text-xs font-medium text-indigo-600 hover:text-indigo-500">
          View All
        </Link>
      </div>

      <ul className="mt-4 space-y-3">
        {data.items.map((item) => {
          const Icon = EVENT_ICONS[item.event_type] ?? BoltIcon;
          const color = EVENT_COLORS[item.event_type] ?? 'text-gray-500';
          return (
            <li key={item.id} className="flex items-start gap-3">
              <Icon className={`mt-0.5 h-5 w-5 shrink-0 ${color}`} />
              <div className="min-w-0 flex-1">
                <p className="text-sm text-gray-700">
                  {item.symbol && <span className="font-medium">{item.symbol} </span>}
                  {item.message}
                </p>
                <p className="mt-0.5 text-xs text-gray-400">{timeAgo(item.created_at)}</p>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
