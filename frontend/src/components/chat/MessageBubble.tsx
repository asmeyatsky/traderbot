import ReactMarkdown from 'react-markdown';
import type { ChatMessage } from '../../types/chat';
import TradeActionCard from './TradeActionCard';

interface MessageBubbleProps {
  message: ChatMessage;
  onConfirmTrade?: (action: ChatMessage['trade_actions'][0]) => void;
}

export default function MessageBubble({ message, onConfirmTrade }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 text-sm ${
          isUser
            ? 'bg-indigo-600 text-white'
            : 'bg-white text-gray-800 shadow-sm ring-1 ring-gray-100'
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none prose-headings:text-gray-800 prose-p:text-gray-700 prose-strong:text-gray-800 prose-code:rounded prose-code:bg-gray-100 prose-code:px-1 prose-code:py-0.5 prose-code:text-indigo-600">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {message.trade_actions.length > 0 && (
          <div className="mt-3 space-y-2">
            {message.trade_actions.map((action, i) => (
              <TradeActionCard
                key={`${action.symbol}-${i}`}
                action={action}
                onConfirm={() => onConfirmTrade?.(action)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
