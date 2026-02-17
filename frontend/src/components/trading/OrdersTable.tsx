import DataTable from '../common/DataTable';
import OrderStatusBadge from './OrderStatusBadge';
import { formatCurrency, formatDateTime } from '../../lib/format';
import { useCancelOrder } from '../../hooks/use-orders';
import type { Order } from '../../types/order';

export default function OrdersTable({ orders }: { orders: Order[] }) {
  const { mutate: cancel } = useCancelOrder();

  return (
    <DataTable
      data={orders}
      keyExtractor={(o) => o.id}
      emptyMessage="No orders yet"
      columns={[
        { key: 'symbol', header: 'Symbol', render: (o) => <span className="font-medium">{o.symbol}</span> },
        {
          key: 'side',
          header: 'Side',
          render: (o) => (
            <span className={o.side === 'BUY' ? 'text-green-600' : 'text-red-600'}>{o.side}</span>
          ),
        },
        { key: 'type', header: 'Type', render: (o) => o.order_type },
        { key: 'qty', header: 'Qty', render: (o) => o.quantity },
        { key: 'price', header: 'Price', render: (o) => (o.price ? formatCurrency(o.price) : 'MKT') },
        { key: 'status', header: 'Status', render: (o) => <OrderStatusBadge status={o.status} /> },
        { key: 'created', header: 'Created', render: (o) => formatDateTime(o.created_at) },
        {
          key: 'actions',
          header: '',
          render: (o) =>
            o.status === 'PENDING' ? (
              <button
                onClick={() => cancel(o.id)}
                className="text-sm text-red-600 hover:text-red-800"
              >
                Cancel
              </button>
            ) : null,
        },
      ]}
    />
  );
}
