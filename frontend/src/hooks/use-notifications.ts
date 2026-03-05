import { useEffect } from 'react';
import { getWebSocketClient, type WSMessage } from '../api/websocket';
import { useNotificationStore } from '../stores/notification-store';
import { useAuthStore } from '../stores/auth-store';

export function useWebSocketNotifications() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const add = useNotificationStore((s) => s.add);

  useEffect(() => {
    if (!isAuthenticated) return;

    const ws = getWebSocketClient();
    ws.connect();

    const unsubs = [
      ws.on('order_update', (msg: WSMessage) => {
        const event = msg.event as string;
        const data = msg.data as Record<string, unknown>;
        const symbol = data?.symbol as string ?? '';
        if (event === 'OrderExecutedEvent') {
          add({ type: 'success', title: 'Order Executed', message: `${symbol} order filled` });
        } else if (event === 'OrderPlacedEvent') {
          add({ type: 'info', title: 'Order Placed', message: `${symbol} order submitted` });
        } else if (event === 'OrderCancelledEvent') {
          add({ type: 'warning', title: 'Order Cancelled', message: `${symbol} order cancelled` });
        }
      }),
      ws.on('auto_trade', (msg: WSMessage) => {
        const data = msg.data as Record<string, unknown>;
        const symbol = data?.symbol as string ?? '';
        const signal = data?.signal as string ?? '';
        add({
          type: 'info',
          title: 'Auto-Trade',
          message: `${signal} signal for ${symbol}`,
          duration: 8000,
        });
      }),
      ws.on('risk_alert', (msg: WSMessage) => {
        const data = msg.data as Record<string, unknown>;
        const message = data?.message as string ?? 'Risk limit reached';
        add({ type: 'warning', title: 'Risk Alert', message, duration: 10000 });
      }),
    ];

    return () => {
      unsubs.forEach((unsub) => unsub());
    };
  }, [isAuthenticated, add]);
}
