import { useAuthStore } from '../stores/auth-store';
import { useDashboardOverview } from '../hooks/use-dashboard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import PortfolioSummaryCard from '../components/dashboard/PortfolioSummaryCard';
import PnLChart from '../components/dashboard/PnLChart';
import AllocationPieChart from '../components/dashboard/AllocationPieChart';
import TopMovers from '../components/dashboard/TopMovers';

export default function DashboardPage() {
  const userId = useAuthStore((s) => s.user?.id);
  const { data, isLoading, error, refetch } = useDashboardOverview(userId);

  if (isLoading) return <LoadingSpinner className="py-20" />;
  if (error) return <ErrorAlert message="Failed to load dashboard" onRetry={() => refetch()} />;
  if (!data) return null;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      <PortfolioSummaryCard data={data} />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <PnLChart data={data.performance_history ?? []} />
        </div>
        <AllocationPieChart data={data.allocation ?? []} />
      </div>
      <TopMovers gainers={data.top_performers ?? []} losers={data.worst_performers ?? []} />
    </div>
  );
}
