import { useAuthStore } from '../stores/auth-store';
import { useDashboardOverview } from '../hooks/use-dashboard';
import { BanknotesIcon } from '@heroicons/react/24/outline';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import EmptyState from '../components/common/EmptyState';
import PortfolioSummaryCard from '../components/dashboard/PortfolioSummaryCard';
import PnLChart from '../components/dashboard/PnLChart';
import AllocationPieChart from '../components/dashboard/AllocationPieChart';
import TopMovers from '../components/dashboard/TopMovers';
import AutoTradingStatusCard from '../components/dashboard/AutoTradingStatusCard';
import RecentActivityFeed from '../components/dashboard/RecentActivityFeed';

export default function DashboardPage() {
  const userId = useAuthStore((s) => s.user?.id);
  const { data, isLoading, error, refetch } = useDashboardOverview(userId);

  if (isLoading) return <LoadingSpinner className="py-20" />;
  if (error) return <ErrorAlert message="Failed to load dashboard" onRetry={() => refetch()} />;
  if (!data) return null;

  const isEmpty = !data.allocation?.length && !data.performance_history?.length;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <AutoTradingStatusCard />
        <RecentActivityFeed />
      </div>

      {isEmpty ? (
        <EmptyState
          icon={BanknotesIcon}
          title="Your dashboard is waiting"
          description="Fund your account and place your first trade to see portfolio performance, allocation, and market movers."
          ctaLabel="Fund Your Account"
          ctaTo="/portfolio"
        />
      ) : (
        <>
          <PortfolioSummaryCard data={data} />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <PnLChart data={data.performance_history ?? []} />
            </div>
            <AllocationPieChart data={data.allocation ?? []} />
          </div>
          <TopMovers gainers={data.top_performers ?? []} losers={data.worst_performers ?? []} />
        </>
      )}
    </div>
  );
}
