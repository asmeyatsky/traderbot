import { useEffect, useMemo, useState } from 'react';
import { useChatStore } from '../../stores/chat-store';
import {
  useConversations,
  useConversation,
  useCreateConversation,
  useDeleteConversation,
  useSendMessage,
} from '../../hooks/use-chat';
import { useCreateOrder } from '../../hooks/use-orders';
import { extractDisciplineVeto } from '../../api/orders';
import ConversationList from './ConversationList';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import DisciplineVetoModal from './DisciplineVetoModal';
import type { TradeAction, ChatMessage } from '../../types/chat';
import type { CreateOrderRequest, DisciplineVeto } from '../../types/order';

const EMPTY_MESSAGES: ChatMessage[] = [];

export default function ChatView() {
  const conversations = useChatStore((s) => s.conversations);
  const activeConversationId = useChatStore((s) => s.activeConversationId);
  const streamingContent = useChatStore((s) => s.streamingContent);
  const isStreaming = useChatStore((s) => s.isStreaming);
  const setActiveConversation = useChatStore((s) => s.setActiveConversation);

  const allPendingMessages = useChatStore((s) => s.pendingMessages);
  const pendingMessages = useMemo(
    () => (activeConversationId ? (allPendingMessages[activeConversationId] ?? EMPTY_MESSAGES) : EMPTY_MESSAGES),
    [activeConversationId, allPendingMessages],
  );
  const executedTradeKeys = useChatStore((s) => s.executedTradeKeys);
  const markTradeExecuted = useChatStore((s) => s.markTradeExecuted);

  const { data: convData } = useConversations();
  const { data: activeConv } = useConversation(activeConversationId);
  const { mutate: createConv } = useCreateConversation();
  const { mutate: deleteConv } = useDeleteConversation();
  const { sendMessage } = useSendMessage();
  const { mutate: createOrder, isPending: isOrderPending } = useCreateOrder();

  // Auto-create a conversation if none exist
  useEffect(() => {
    if (convData && convData.conversations.length === 0) {
      createConv({});
    }
  }, [convData, createConv]);

  // Auto-select first conversation
  useEffect(() => {
    if (!activeConversationId && conversations.length > 0) {
      setActiveConversation(conversations[0].id);
    }
  }, [conversations, activeConversationId, setActiveConversation]);

  // Merge server messages with pending messages (deduped by id)
  const serverMessages = activeConv?.messages ?? [];
  const serverIds = new Set(serverMessages.map((m) => m.id));
  const messages = [...serverMessages, ...pendingMessages.filter((m) => !serverIds.has(m.id))];

  const handleSend = (content: string) => {
    if (!activeConversationId) return;
    sendMessage(activeConversationId, content);
  };

  const handleNewChat = () => {
    createConv({});
  };

  // Phase 10.1 — veto modal state. When the backend refuses a trade with
  // a discipline_veto body, we pause the order, show the rules, and offer
  // the user an explicit override path. Cancel just drops the request.
  const [vetoState, setVetoState] = useState<
    { pendingOrder: CreateOrderRequest; tradeKey: string; vetoes: DisciplineVeto[] } | null
  >(null);

  const submitOrder = (payload: CreateOrderRequest, tradeKey: string) => {
    if (!activeConversationId) return;
    createOrder(payload, {
      onSuccess: () => {
        markTradeExecuted(tradeKey);
        setVetoState(null);
      },
      onError: (err) => {
        const veto = extractDisciplineVeto(err);
        if (veto) {
          setVetoState({ pendingOrder: payload, tradeKey, vetoes: veto.vetoes });
          return;
        }
        sendMessage(
          activeConversationId,
          `Order failed: ${err instanceof Error ? err.message : 'Unknown error'}`,
        );
      },
    });
  };

  const handleConfirmTrade = (action: TradeAction) => {
    if (!activeConversationId) return;
    const tradeKey = `${action.symbol}-${action.action}-${action.quantity}`;
    submitOrder(
      {
        symbol: action.symbol,
        position_type: action.action === 'BUY' ? 'LONG' : 'SHORT',
        order_type: 'MARKET',
        quantity: action.quantity,
      },
      tradeKey,
    );
  };

  const handleVetoOverride = () => {
    if (!vetoState) return;
    submitOrder(
      { ...vetoState.pendingOrder, override_discipline_vetoes: true },
      vetoState.tradeKey,
    );
  };

  const handleVetoCancel = () => setVetoState(null);

  return (
    <div className="flex h-full">
      {/* Conversation list — hidden on mobile, shown on desktop */}
      <div className="hidden w-64 md:block">
        <ConversationList
          conversations={conversations}
          activeId={activeConversationId}
          onSelect={setActiveConversation}
          onNew={handleNewChat}
          onDelete={(id) => deleteConv(id)}
        />
      </div>

      {/* Main chat area */}
      <div className="flex flex-1 flex-col">
        <MessageList
          messages={messages}
          streamingContent={streamingContent}
          isStreaming={isStreaming}
          onConfirmTrade={handleConfirmTrade}
          onSuggestionClick={handleSend}
          executedTradeKeys={executedTradeKeys}
          isOrderPending={isOrderPending}
        />
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
      <DisciplineVetoModal
        open={vetoState !== null}
        vetoes={vetoState?.vetoes ?? []}
        isOverriding={isOrderPending}
        onOverride={handleVetoOverride}
        onCancel={handleVetoCancel}
      />
    </div>
  );
}
