import { useEffect } from 'react';
import { useChatStore } from '../../stores/chat-store';
import {
  useConversations,
  useConversation,
  useCreateConversation,
  useDeleteConversation,
  useSendMessage,
} from '../../hooks/use-chat';
import ConversationList from './ConversationList';
import MessageList from './MessageList';
import ChatInput from './ChatInput';

export default function ChatView() {
  const {
    conversations,
    activeConversationId,
    streamingContent,
    isStreaming,
    setActiveConversation,
  } = useChatStore();

  const { data: convData } = useConversations();
  const { data: activeConv } = useConversation(activeConversationId);
  const { mutate: createConv } = useCreateConversation();
  const { mutate: deleteConv } = useDeleteConversation();
  const { sendMessage } = useSendMessage();

  // Auto-create a conversation if none exist
  useEffect(() => {
    if (convData && convData.conversations.length === 0) {
      createConv({});
    }
  }, [convData]);

  // Auto-select first conversation
  useEffect(() => {
    if (!activeConversationId && conversations.length > 0) {
      setActiveConversation(conversations[0].id);
    }
  }, [conversations, activeConversationId]);

  const messages = activeConv?.messages ?? [];

  const handleSend = (content: string) => {
    if (!activeConversationId) return;
    sendMessage(activeConversationId, content);
  };

  const handleNewChat = () => {
    createConv({});
  };

  const handleConfirmTrade = (action: { symbol: string; action: string; quantity: number }) => {
    if (!activeConversationId) return;
    sendMessage(
      activeConversationId,
      `Yes, confirm the ${action.action} order for ${action.quantity} shares of ${action.symbol}.`,
    );
  };

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
        />
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
