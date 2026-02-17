const statusStyles: Record<string, string> = {
  PENDING: 'bg-yellow-100 text-yellow-800',
  FILLED: 'bg-green-100 text-green-800',
  PARTIALLY_FILLED: 'bg-blue-100 text-blue-800',
  CANCELLED: 'bg-gray-100 text-gray-800',
  REJECTED: 'bg-red-100 text-red-800',
};

export default function OrderStatusBadge({ status }: { status: string }) {
  return (
    <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${statusStyles[status] ?? 'bg-gray-100 text-gray-800'}`}>
      {status}
    </span>
  );
}
