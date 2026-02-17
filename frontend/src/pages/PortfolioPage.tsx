import { usePortfolio, usePortfolioAllocation } from '../hooks/use-portfolio';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import StatCard from '../components/common/StatCard';
import PositionsTable from '../components/portfolio/PositionsTable';
import AllocationChart from '../components/portfolio/AllocationChart';
import CashManagement from '../components/portfolio/CashManagement';
import { formatCurrency } from '../lib/format';

export default function PortfolioPage() {
  const { data: portfolio, isLoading, error, refetch } = usePortfolio();
  const { data: allocation } = usePortfolioAllocation();

  if (isLoading) return <LoadingSpinner className="py-20" />;
  if (error) return <ErrorAlert message="Failed to load portfolio" onRetry={() => refetch()} />;
  if (!portfolio) return null;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Portfolio</h1>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard title="Total Value" value={formatCurrency(portfolio.total_value)} />
        <StatCard title="Cash Balance" value={formatCurrency(portfolio.cash_balance)} />
        <StatCard title="Positions" value={String(portfolio.positions?.length ?? 0)} />
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="rounded-lg bg-white shadow-sm ring-1 ring-gray-900/5">
            <div className="border-b border-gray-200 px-4 py-3">
              <h3 className="text-sm font-medium text-gray-700">Positions</h3>
            </div>
            <PositionsTable positions={portfolio.positions ?? []} />
          </div>
        </div>
        <div className="space-y-6">
          <CashManagement cashBalance={portfolio.cash_balance} totalValue={portfolio.total_value} />
          {allocation && <AllocationChart data={allocation.allocations ?? []} />}
        </div>
      </div>
    </div>
  );
}
