import StatCard from '../common/StatCard';
import { formatCurrency, formatPercent } from '../../lib/format';
import type { DashboardOverview } from '../../types/dashboard';

export default function PortfolioSummaryCard({ data }: { data: DashboardOverview }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard
        title="Portfolio Value"
        value={formatCurrency(data.portfolio_value)}
        change={formatPercent(data.total_pnl_percent)}
        changeType={data.total_pnl_percent >= 0 ? 'positive' : 'negative'}
      />
      <StatCard
        title="Daily P&L"
        value={formatCurrency(data.daily_pnl)}
        change={formatPercent(data.daily_pnl_percent)}
        changeType={data.daily_pnl_percent >= 0 ? 'positive' : 'negative'}
      />
      <StatCard
        title="Total P&L"
        value={formatCurrency(data.total_pnl)}
        changeType={data.total_pnl >= 0 ? 'positive' : 'negative'}
      />
      <StatCard title="Positions" value={String(data.positions_count)} />
    </div>
  );
}
