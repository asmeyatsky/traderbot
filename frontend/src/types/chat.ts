export interface TradeAction {
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  reasoning: string;
  confidence: number;
  executed: boolean;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: string;
  trade_actions: TradeAction[];
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  messages: ChatMessage[];
}

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
}

export interface CreateConversationRequest {
  title?: string;
}

export interface SendMessageRequest {
  content: string;
}

export interface SSEEvent {
  type: 'text_delta' | 'tool_call' | 'tool_result' | 'done' | 'error';
  content?: string;
  tool_call?: {
    id: string;
    name: string;
    arguments: Record<string, unknown>;
  };
  tool_result?: {
    tool_call_id: string;
    content: string;
    is_error: boolean;
  };
  metadata?: Record<string, unknown>;
}
