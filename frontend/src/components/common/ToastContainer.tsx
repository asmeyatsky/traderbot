import { useNotificationStore, type NotificationType } from '../../stores/notification-store';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

const icons: Record<NotificationType, typeof CheckCircleIcon> = {
  success: CheckCircleIcon,
  error: XCircleIcon,
  warning: ExclamationTriangleIcon,
  info: InformationCircleIcon,
};

const colors: Record<NotificationType, string> = {
  success: 'text-green-500',
  error: 'text-red-500',
  warning: 'text-amber-500',
  info: 'text-indigo-500',
};

export default function ToastContainer() {
  const notifications = useNotificationStore((s) => s.notifications);
  const dismiss = useNotificationStore((s) => s.dismiss);

  if (notifications.length === 0) return null;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-[100] flex flex-col items-center gap-2 px-4">
      {notifications.map((n) => {
        const Icon = icons[n.type];
        return (
          <div
            key={n.id}
            className="pointer-events-auto w-full max-w-sm animate-fade-in rounded-lg bg-white p-4 shadow-lg ring-1 ring-gray-900/10 dark:bg-gray-800 dark:ring-white/10"
          >
            <div className="flex items-start gap-3">
              <Icon className={`h-5 w-5 shrink-0 ${colors[n.type]}`} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{n.title}</p>
                {n.message && (
                  <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400">{n.message}</p>
                )}
              </div>
              <button
                onClick={() => dismiss(n.id)}
                className="shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <XMarkIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
