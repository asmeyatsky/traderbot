import { useEffect, useRef } from 'react';
import type { ChatMessage, TradeAction } from '../../types/chat';
import MessageBubble from './MessageBubble';
import StreamingIndicator from './StreamingIndicator';
import ReactMarkdown from 'react-markdown';

interface MessageListProps {
  messages: ChatMessage[];
  streamingContent: string;
  isStreaming: boolean;
  onConfirmTrade?: (action: TradeAction) => void;
  onSuggestionClick?: (text: string) => void;
  executedTradeKeys?: string[];
  isOrderPending?: boolean;
}

export default function MessageList({
  messages,
  streamingContent,
  isStreaming,
  onConfirmTrade,
  onSuggestionClick,
  executedTradeKeys = [],
  isOrderPending = false,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length, streamingContent]);

  if (messages.length === 0 && !isStreaming) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center p-8 text-center">
        <div className="mb-4 text-4xl">&#x1f4c8;</div>
        <h2 className="text-lg font-semibold text-gray-800 dark:text-white">
          What would you like to know?
        </h2>
        <p className="mt-2 max-w-sm text-sm text-gray-500 dark:text-gray-400">
          Ask about stock prices, get ML predictions, analyze your portfolio, or
          explore trading ideas.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {[
            "What's the price of AAPL?",
            'Show me my portfolio',
            'Any oversold tech stocks?',
            'Should I buy TSLA?',
          ].map((suggestion) => (
            <button
              key={suggestion}
              className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-600 transition-colors hover:border-indigo-300 hover:text-indigo-600 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-400 dark:hover:border-indigo-500 dark:hover:text-indigo-400"
              onClick={() => onSuggestionClick?.(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          onConfirmTrade={onConfirmTrade}
          executedTradeKeys={executedTradeKeys}
          isOrderPending={isOrderPending}
        />
      ))}

      {isStreaming && streamingContent && (
        <div className="flex justify-start">
          <div className="max-w-[85%] sm:max-w-[75%] rounded-2xl bg-white px-4 py-3 text-sm shadow-sm ring-1 ring-gray-100 dark:bg-gray-800 dark:ring-gray-700">
            <div className="prose prose-sm max-w-none dark:prose-p:text-gray-300">
              <ReactMarkdown>{streamingContent}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {isStreaming && !streamingContent && <StreamingIndicator />}

      <div ref={bottomRef} />
    </div>
  );
}
