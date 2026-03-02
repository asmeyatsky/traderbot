import { useEffect, useRef } from 'react';
import { getWebSocketClient, type WSMessage } from '../api/websocket';
import { useAuthStore } from '../stores/auth-store';

/**
 * Connect to WebSocket on mount, disconnect on unmount.
 */
export function useWebSocket() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const client = getWebSocketClient();

  useEffect(() => {
    if (isAuthenticated) {
      client.connect();
    }
    return () => client.disconnect();
  }, [isAuthenticated]);

  return client;
}

/**
 * Subscribe to a specific message type.
 */
export function useWSMessage(type: string, handler: (msg: WSMessage) => void) {
  const client = getWebSocketClient();
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    return client.on(type, (msg) => handlerRef.current(msg));
  }, [type]);
}

/**
 * Subscribe to a price channel.
 */
export function usePriceUpdates(symbol: string | null) {
  const client = getWebSocketClient();

  useEffect(() => {
    if (!symbol) return;
    client.subscribe(`prices:${symbol}`);
    return () => client.unsubscribe(`prices:${symbol}`);
  }, [symbol]);
}
