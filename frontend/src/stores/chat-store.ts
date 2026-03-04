import { create } from 'zustand';
import type { Conversation, ChatMessage } from '../types/chat';

interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  streamingContent: string;
  isStreaming: boolean;
  pendingMessages: Record<string, ChatMessage[]>;
  executedTradeKeys: string[];

  setConversations: (conversations: Conversation[]) => void;
  setActiveConversation: (id: string | null) => void;
  addConversation: (conversation: Conversation) => void;
  removeConversation: (id: string) => void;
  addMessage: (conversationId: string, message: ChatMessage) => void;
  addPendingMessage: (conversationId: string, message: ChatMessage) => void;
  clearPendingMessages: (conversationId: string) => void;
  markTradeExecuted: (key: string) => void;
  setStreamingContent: (content: string) => void;
  appendStreamingContent: (delta: string) => void;
  setIsStreaming: (streaming: boolean) => void;
  clearStreaming: () => void;
}

export const useChatStore = create<ChatState>((set) => ({
  conversations: [],
  activeConversationId: null,
  streamingContent: '',
  isStreaming: false,
  pendingMessages: {},
  executedTradeKeys: [],

  setConversations: (conversations) => set({ conversations }),

  setActiveConversation: (id) => set({ activeConversationId: id }),

  addConversation: (conversation) =>
    set((state) => ({
      conversations: [conversation, ...state.conversations],
    })),

  removeConversation: (id) =>
    set((state) => ({
      conversations: state.conversations.filter((c) => c.id !== id),
      activeConversationId:
        state.activeConversationId === id ? null : state.activeConversationId,
    })),

  addMessage: (conversationId, message) =>
    set((state) => ({
      conversations: state.conversations.map((c) =>
        c.id === conversationId
          ? { ...c, messages: [...c.messages, message], message_count: c.message_count + 1 }
          : c,
      ),
    })),

  addPendingMessage: (conversationId, message) =>
    set((state) => ({
      pendingMessages: {
        ...state.pendingMessages,
        [conversationId]: [...(state.pendingMessages[conversationId] ?? []), message],
      },
    })),

  clearPendingMessages: (conversationId) =>
    set((state) => {
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { [conversationId]: _removed, ...rest } = state.pendingMessages;
      return { pendingMessages: rest };
    }),

  markTradeExecuted: (key) =>
    set((state) => ({
      executedTradeKeys: [...state.executedTradeKeys, key],
    })),

  setStreamingContent: (content) => set({ streamingContent: content }),
  appendStreamingContent: (delta) =>
    set((state) => ({ streamingContent: state.streamingContent + delta })),
  setIsStreaming: (streaming) => set({ isStreaming: streaming }),
  clearStreaming: () => set({ streamingContent: '', isStreaming: false }),
}));
