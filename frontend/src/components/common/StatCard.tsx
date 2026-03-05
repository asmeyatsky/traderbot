interface StatCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon?: React.ReactNode;
}

export default function StatCard({ title, value, change, changeType = 'neutral', icon }: StatCardProps) {
  const changeColor =
    changeType === 'positive'
      ? 'text-green-600 dark:text-green-400'
      : changeType === 'negative'
        ? 'text-red-600 dark:text-red-400'
        : 'text-gray-500 dark:text-gray-400';

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5 dark:bg-gray-800 dark:ring-white/10">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
        {icon && <div className="text-gray-400 dark:text-gray-500">{icon}</div>}
      </div>
      <p className="mt-2 text-3xl font-semibold text-gray-900 dark:text-white">{value}</p>
      {change && <p className={`mt-1 text-sm ${changeColor}`}>{change}</p>}
    </div>
  );
}
