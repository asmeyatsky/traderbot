import apiClient from './client';
import type {
  Conversation,
  ConversationListResponse,
  CreateConversationRequest,
  SSEEvent,
} from '../types/chat';
import { API_BASE_URL } from '../lib/constants';
import { useAuthStore } from '../stores/auth-store';

const PREFIX = '/chat';

export async function createConversation(
  req: CreateConversationRequest = {},
): Promise<Conversation> {
  const { data } = await apiClient.post(`${PREFIX}/conversations`, req);
  return data;
}

export async function listConversations(
  limit = 50,
  offset = 0,
): Promise<ConversationListResponse> {
  const { data } = await apiClient.get(`${PREFIX}/conversations`, {
    params: { limit, offset },
  });
  return data;
}

export async function getConversation(id: string): Promise<Conversation> {
  const { data } = await apiClient.get(`${PREFIX}/conversations/${id}`);
  return data;
}

export async function deleteConversation(id: string): Promise<void> {
  await apiClient.delete(`${PREFIX}/conversations/${id}`);
}

/**
 * Send a message and receive SSE stream.
 * Returns an async generator that yields SSEEvent objects.
 */
export async function* sendMessageStream(
  conversationId: string,
  content: string,
): AsyncGenerator<SSEEvent> {
  const token = useAuthStore.getState().token;
  const url = `${API_BASE_URL}${PREFIX}/conversations/${conversationId}/messages`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ content }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event: SSEEvent = JSON.parse(line.slice(6));
          yield event;
        } catch (e) {
          console.warn('Malformed SSE event, skipping:', line.slice(6, 100), e);
        }
      }
    }
  }
}
