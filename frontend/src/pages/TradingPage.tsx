import { useState } from 'react';
import { useOrders } from '../hooks/use-orders';
import OrderEntryForm from '../components/trading/OrderEntryForm';
import OrdersTable from '../components/trading/OrdersTable';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';
import { ORDER_STATUSES } from '../lib/constants';

export default function TradingPage() {
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const { data, isLoading, error, refetch } = useOrders(statusFilter);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Trading</h1>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div>
          <OrderEntryForm />
        </div>
        <div className="lg:col-span-2">
          <div className="rounded-lg bg-white shadow-sm ring-1 ring-gray-900/5">
            <div className="border-b border-gray-200 px-4 py-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-700">Filter:</span>
                <button
                  onClick={() => setStatusFilter(undefined)}
                  className={`rounded-full px-3 py-1 text-xs font-medium ${!statusFilter ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                >
                  All
                </button>
                {ORDER_STATUSES.map((s) => (
                  <button
                    key={s}
                    onClick={() => setStatusFilter(s)}
                    className={`rounded-full px-3 py-1 text-xs font-medium ${statusFilter === s ? 'bg-indigo-100 text-indigo-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            {isLoading ? (
              <LoadingSpinner className="py-10" />
            ) : error ? (
              <div className="p-4">
                <ErrorAlert message="Failed to load orders" onRetry={() => refetch()} />
              </div>
            ) : (
              <OrdersTable orders={data?.orders ?? []} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
