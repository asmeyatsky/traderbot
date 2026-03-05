import { create } from 'zustand';

export type NotificationType = 'success' | 'error' | 'info' | 'warning';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number;
}

interface NotificationState {
  notifications: Notification[];
  add: (n: Omit<Notification, 'id'>) => void;
  dismiss: (id: string) => void;
}

let counter = 0;

export const useNotificationStore = create<NotificationState>()((set) => ({
  notifications: [],
  add: (n) => {
    const id = `notif-${++counter}`;
    set((s) => ({ notifications: [...s.notifications, { ...n, id }] }));
    const duration = n.duration ?? 5000;
    if (duration > 0) {
      setTimeout(() => {
        set((s) => ({ notifications: s.notifications.filter((x) => x.id !== id) }));
      }, duration);
    }
  },
  dismiss: (id) => set((s) => ({ notifications: s.notifications.filter((x) => x.id !== id) })),
}));
