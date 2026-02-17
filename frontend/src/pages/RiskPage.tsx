import { useAuthStore } from '../stores/auth-store';
import { usePortfolioRisk } from '../hooks/use-risk';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import RiskMetricsCard from '../components/risk/RiskMetricsCard';
import StressTestPanel from '../components/risk/StressTestPanel';

export default function RiskPage() {
  const userId = useAuthStore((s) => s.user?.id);
  const { data, isLoading, error, refetch } = usePortfolioRisk(userId);

  if (isLoading) return <LoadingSpinner className="py-20" />;
  if (error) return <ErrorAlert message="Failed to load risk metrics" onRetry={() => refetch()} />;
  if (!data) return <p className="py-10 text-center text-gray-500">No risk data available</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Risk Analytics</h1>
      <RiskMetricsCard data={data} />
      <StressTestPanel results={data.stress_results ?? []} />
    </div>
  );
}
