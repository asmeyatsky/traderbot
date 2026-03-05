import { useState } from 'react';
import { TrophyIcon, UserGroupIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import {
  useLeaderboard,
  useMarketplace,
  useMyStrategies,
  useCreateStrategy,
  useFollowStrategy,
  useUnfollowStrategy,
  useForkStrategy,
  useDeleteStrategy,
  useSaveBacktestResult,
  type Strategy,
  type LeaderboardEntry,
} from '../hooks/use-strategies';
import { useRunBacktest } from '../hooks/use-backtest';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useNotificationStore } from '../stores/notification-store';

type Tab = 'leaderboard' | 'marketplace' | 'my-strategies';

export default function LeaderboardPage() {
  const [tab, setTab] = useState<Tab>('leaderboard');

  return (
    <div className="h-full overflow-auto p-4 sm:p-6">
      <h1 className="text-lg font-semibold text-gray-800 dark:text-white">Strategy Hub</h1>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
        Discover top strategies, share yours, and copy trade the best performers.
      </p>

      <div className="mt-4 flex gap-2">
        {([
          ['leaderboard', 'Leaderboard'],
          ['marketplace', 'Marketplace'],
          ['my-strategies', 'My Strategies'],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`rounded-full px-4 py-1.5 text-xs font-medium transition-colors ${
              tab === key
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {tab === 'leaderboard' && <LeaderboardTab />}
        {tab === 'marketplace' && <MarketplaceTab />}
        {tab === 'my-strategies' && <MyStrategiesTab />}
      </div>
    </div>
  );
}

function LeaderboardTab() {
  const { data: entries, isLoading } = useLeaderboard();
  const follow = useFollowStrategy();
  const unfollow = useUnfollowStrategy();

  if (isLoading) return <LoadingSpinner className="py-12" />;
  if (!entries?.length) return <EmptyState icon={TrophyIcon} text="No strategies on the leaderboard yet. Save and publish a strategy to appear here." />;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="pb-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">#</th>
            <th className="pb-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Strategy</th>
            <th className="pb-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Author</th>
            <th className="pb-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400">Return</th>
            <th className="pb-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400">Sharpe</th>
            <th className="pb-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400">Win Rate</th>
            <th className="pb-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400">Drawdown</th>
            <th className="pb-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400">Followers</th>
            <th className="pb-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400"></th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <LeaderboardRow
              key={entry.strategy_id}
              entry={entry}
              onFollow={() => follow.mutate(entry.strategy_id)}
              onUnfollow={() => unfollow.mutate(entry.strategy_id)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LeaderboardRow({ entry, onFollow, onUnfollow }: { entry: LeaderboardEntry; onFollow: () => void; onUnfollow: () => void }) {
  const isPositive = entry.total_return_pct >= 0;
  return (
    <tr className="border-b border-gray-100 dark:border-gray-800">
      <td className="py-3 text-gray-500 dark:text-gray-400">{entry.rank}</td>
      <td className="py-3">
        <p className="font-medium text-gray-900 dark:text-white">{entry.strategy_name}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400">{entry.symbol} &middot; {entry.strategy_type.replace('_', ' ')}</p>
      </td>
      <td className="py-3 text-gray-600 dark:text-gray-400">{entry.author_name}</td>
      <td className={`py-3 text-right font-semibold ${isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
        {isPositive ? '+' : ''}{entry.total_return_pct}%
      </td>
      <td className="py-3 text-right text-gray-700 dark:text-gray-300">{entry.sharpe_ratio.toFixed(2)}</td>
      <td className="py-3 text-right text-gray-700 dark:text-gray-300">{entry.win_rate}%</td>
      <td className="py-3 text-right text-gray-700 dark:text-gray-300">{entry.max_drawdown_pct}%</td>
      <td className="py-3 text-right">
        <span className="inline-flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
          <UserGroupIcon className="h-3.5 w-3.5" />
          {entry.follower_count}
        </span>
      </td>
      <td className="py-3 text-right">
        <button
          onClick={entry.is_following ? onUnfollow : onFollow}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            entry.is_following
              ? 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
        >
          {entry.is_following ? 'Following' : 'Follow'}
        </button>
      </td>
    </tr>
  );
}

function MarketplaceTab() {
  const { data: strategies, isLoading } = useMarketplace();
  const fork = useForkStrategy();
  const follow = useFollowStrategy();
  const unfollow = useUnfollowStrategy();
  const addNotif = useNotificationStore((s) => s.add);

  if (isLoading) return <LoadingSpinner className="py-12" />;
  if (!strategies?.length) return <EmptyState icon={TrophyIcon} text="No public strategies yet. Create one and make it public to share with the community." />;

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {strategies.map((s) => (
        <StrategyCard
          key={s.id}
          strategy={s}
          onFollow={() => (s.is_following ? unfollow : follow).mutate(s.id)}
          onFork={() => {
            fork.mutate(s.id, {
              onSuccess: () => addNotif({ type: 'success', title: 'Strategy forked', message: `"${s.name}" copied to your strategies` }),
            });
          }}
        />
      ))}
    </div>
  );
}

function StrategyCard({ strategy, onFollow, onFork }: { strategy: Strategy; onFollow: () => void; onFork: () => void }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-semibold text-gray-900 dark:text-white">{strategy.name}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">by {strategy.author_name}</p>
        </div>
        <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600 dark:bg-gray-700 dark:text-gray-300">
          {strategy.strategy_type.replace('_', ' ')}
        </span>
      </div>
      {strategy.description && (
        <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 line-clamp-2">{strategy.description}</p>
      )}
      <div className="mt-3 flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
        <span>{strategy.symbol}</span>
        {strategy.best_return_pct != null && (
          <span className={strategy.best_return_pct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
            {strategy.best_return_pct >= 0 ? '+' : ''}{strategy.best_return_pct}%
          </span>
        )}
        <span className="flex items-center gap-0.5">
          <UserGroupIcon className="h-3 w-3" />
          {strategy.follower_count}
        </span>
      </div>
      <div className="mt-3 flex gap-2">
        <button
          onClick={onFollow}
          className={`flex-1 rounded-md py-1.5 text-xs font-medium transition-colors ${
            strategy.is_following
              ? 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
        >
          {strategy.is_following ? 'Following' : 'Follow'}
        </button>
        <button
          onClick={onFork}
          className="flex items-center gap-1 rounded-md bg-gray-100 px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
        >
          <ArrowPathIcon className="h-3.5 w-3.5" />
          Fork
        </button>
      </div>
    </div>
  );
}

function MyStrategiesTab() {
  const { data: strategies, isLoading } = useMyStrategies();
  const createStrategy = useCreateStrategy();
  const deleteStrategy = useDeleteStrategy();
  const runBacktest = useRunBacktest();
  const saveResult = useSaveBacktestResult();
  const addNotif = useNotificationStore((s) => s.add);

  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [strategyType, setStrategyType] = useState('sma_crossover');
  const [symbol, setSymbol] = useState('AAPL');
  const [isPublic, setIsPublic] = useState(false);

  if (isLoading) return <LoadingSpinner className="py-12" />;

  function handleCreate() {
    if (!name.trim()) return;
    createStrategy.mutate(
      { name: name.trim(), description, strategy_type: strategyType, symbol: symbol.toUpperCase(), is_public: isPublic },
      {
        onSuccess: () => {
          setShowCreate(false);
          setName('');
          setDescription('');
          addNotif({ type: 'success', title: 'Strategy created' });
        },
      },
    );
  }

  function handleRunAndSave(strategy: Strategy) {
    runBacktest.mutate(
      { strategy: strategy.strategy_type, symbol: strategy.symbol, initial_capital: 10000 },
      {
        onSuccess: (result) => {
          if (result && !result.error) {
            saveResult.mutate({
              strategy_id: strategy.id,
              symbol: strategy.symbol,
              initial_capital: result.initial_capital,
              final_value: result.final_value,
              total_return_pct: result.total_return_pct,
              sharpe_ratio: result.sharpe_ratio,
              max_drawdown_pct: result.max_drawdown_pct,
              win_rate: result.win_rate,
              total_trades: result.total_trades,
              volatility: result.volatility,
              profit_factor: result.profit_factor,
            }, {
              onSuccess: () => addNotif({ type: 'success', title: 'Backtest saved', message: `${result.total_return_pct}% return` }),
            });
          }
        },
      },
    );
  }

  return (
    <div>
      <button
        onClick={() => setShowCreate(true)}
        className="mb-4 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700"
      >
        + New Strategy
      </button>

      {showCreate && (
        <div className="mb-6 rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">Create Strategy</h3>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Strategy name"
              className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
            />
            <select
              value={strategyType}
              onChange={(e) => setStrategyType(e.target.value)}
              className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
            >
              <option value="sma_crossover">SMA Crossover</option>
              <option value="rsi_mean_reversion">RSI Mean Reversion</option>
              <option value="momentum">Momentum Breakout</option>
            </select>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="Symbol (e.g. AAPL)"
              className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
            />
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Description (optional)"
              className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
            />
          </div>
          <div className="mt-3 flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
              <input
                type="checkbox"
                checked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
                className="h-4 w-4 rounded border-gray-300 text-indigo-600 dark:border-gray-600"
              />
              Make public (visible on marketplace)
            </label>
          </div>
          <div className="mt-3 flex gap-2">
            <button onClick={handleCreate} disabled={createStrategy.isPending} className="rounded-md bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
              {createStrategy.isPending ? 'Creating...' : 'Create'}
            </button>
            <button onClick={() => setShowCreate(false)} className="rounded-md bg-gray-100 px-4 py-1.5 text-sm font-medium text-gray-700 dark:bg-gray-700 dark:text-gray-300">Cancel</button>
          </div>
        </div>
      )}

      {!strategies?.length && !showCreate && (
        <EmptyState icon={TrophyIcon} text="No saved strategies yet. Create one to get started." />
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {strategies?.map((s) => (
          <div key={s.id} className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-semibold text-gray-900 dark:text-white">{s.name}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">{s.symbol} &middot; {s.strategy_type.replace('_', ' ')}</p>
              </div>
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${s.is_public ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'}`}>
                {s.is_public ? 'Public' : 'Private'}
              </span>
            </div>
            {s.best_return_pct != null && (
              <p className={`mt-2 text-sm font-semibold ${s.best_return_pct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                Best: {s.best_return_pct >= 0 ? '+' : ''}{s.best_return_pct}%
                {s.best_sharpe != null && <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">Sharpe {s.best_sharpe.toFixed(2)}</span>}
              </p>
            )}
            <div className="mt-3 flex gap-2">
              <button
                onClick={() => handleRunAndSave(s)}
                disabled={runBacktest.isPending}
                className="flex-1 rounded-md bg-indigo-600 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
              >
                {runBacktest.isPending ? 'Running...' : 'Run Backtest'}
              </button>
              <button
                onClick={() => {
                  if (confirm(`Delete "${s.name}"?`)) deleteStrategy.mutate(s.id);
                }}
                className="rounded-md bg-red-50 px-3 py-1.5 text-xs font-medium text-red-600 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState({ icon: Icon, text }: { icon: typeof TrophyIcon; text: string }) {
  return (
    <div className="flex flex-col items-center py-12 text-center">
      <Icon className="h-12 w-12 text-gray-300 dark:text-gray-600" />
      <p className="mt-4 max-w-sm text-sm text-gray-500 dark:text-gray-400">{text}</p>
    </div>
  );
}
