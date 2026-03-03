import { describe, it, expect, vi, beforeAll } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MessageList from './MessageList';
import ChatInput from './ChatInput';
import type { ChatMessage } from '../../types/chat';

// jsdom does not implement scrollIntoView
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

// Mock react-markdown to avoid ESM/transform issues in test env
vi.mock('react-markdown', () => ({
  default: ({ children }: { children: string }) => <span>{children}</span>,
}));

vi.mock('./StreamingIndicator', () => ({
  default: () => <div data-testid="streaming-indicator">Thinking...</div>,
}));

vi.mock('./TradeActionCard', () => ({
  default: ({
    action,
    onConfirm,
  }: {
    action: { symbol: string; action: string };
    onConfirm: () => void;
  }) => (
    <button data-testid={`trade-action-${action.symbol}`} onClick={onConfirm}>
      {action.action} {action.symbol}
    </button>
  ),
}));

function makeMessage(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: 'msg-1',
    conversation_id: 'conv-1',
    role: 'assistant',
    content: 'Hello, how can I help?',
    created_at: '2026-03-03T12:00:00Z',
    trade_actions: [],
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// MessageList
// ---------------------------------------------------------------------------
describe('MessageList', () => {
  describe('empty state', () => {
    it('renders the empty state heading and description when there are no messages', () => {
      render(
        <MessageList
          messages={[]}
          streamingContent=""
          isStreaming={false}
        />,
      );

      expect(screen.getByText('What would you like to know?')).toBeInTheDocument();
      expect(
        screen.getByText(/Ask about stock prices/),
      ).toBeInTheDocument();
    });

    it('renders all four suggestion buttons', () => {
      render(
        <MessageList
          messages={[]}
          streamingContent=""
          isStreaming={false}
        />,
      );

      const suggestions = [
        "What's the price of AAPL?",
        'Show me my portfolio',
        'Any oversold tech stocks?',
        'Should I buy TSLA?',
      ];

      for (const text of suggestions) {
        expect(screen.getByText(text)).toBeInTheDocument();
      }
    });

    it('calls onSuggestionClick with the suggestion text when a button is clicked', async () => {
      const onSuggestionClick = vi.fn();

      render(
        <MessageList
          messages={[]}
          streamingContent=""
          isStreaming={false}
          onSuggestionClick={onSuggestionClick}
        />,
      );

      await userEvent.click(screen.getByText('Show me my portfolio'));

      expect(onSuggestionClick).toHaveBeenCalledOnce();
      expect(onSuggestionClick).toHaveBeenCalledWith('Show me my portfolio');
    });

    it('does not throw when suggestion is clicked without onSuggestionClick handler', async () => {
      render(
        <MessageList
          messages={[]}
          streamingContent=""
          isStreaming={false}
        />,
      );

      // Should not throw
      await userEvent.click(screen.getByText('Should I buy TSLA?'));
    });
  });

  describe('with messages', () => {
    it('renders user and assistant messages', () => {
      const messages: ChatMessage[] = [
        makeMessage({ id: '1', role: 'user', content: 'What is AAPL at?' }),
        makeMessage({ id: '2', role: 'assistant', content: 'Apple is trading at $185.' }),
      ];

      render(
        <MessageList
          messages={messages}
          streamingContent=""
          isStreaming={false}
        />,
      );

      expect(screen.getByText('What is AAPL at?')).toBeInTheDocument();
      expect(screen.getByText('Apple is trading at $185.')).toBeInTheDocument();
    });

    it('does not show empty state when messages are present', () => {
      render(
        <MessageList
          messages={[makeMessage()]}
          streamingContent=""
          isStreaming={false}
        />,
      );

      expect(screen.queryByText('What would you like to know?')).not.toBeInTheDocument();
    });

    it('renders trade action cards for messages with trade_actions', async () => {
      const onConfirmTrade = vi.fn();
      const message = makeMessage({
        id: '3',
        trade_actions: [
          {
            symbol: 'AAPL',
            action: 'BUY',
            quantity: 10,
            reasoning: 'Strong fundamentals',
            confidence: 0.85,
            executed: false,
          },
        ],
      });

      render(
        <MessageList
          messages={[message]}
          streamingContent=""
          isStreaming={false}
          onConfirmTrade={onConfirmTrade}
        />,
      );

      const tradeButton = screen.getByTestId('trade-action-AAPL');
      expect(tradeButton).toHaveTextContent('BUY AAPL');

      await userEvent.click(tradeButton);
      expect(onConfirmTrade).toHaveBeenCalledOnce();
      expect(onConfirmTrade).toHaveBeenCalledWith(message.trade_actions[0]);
    });
  });

  describe('streaming', () => {
    it('shows the streaming indicator when streaming with no content', () => {
      render(
        <MessageList
          messages={[makeMessage()]}
          streamingContent=""
          isStreaming={true}
        />,
      );

      expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument();
    });

    it('shows streaming content in a bubble when content is available', () => {
      render(
        <MessageList
          messages={[makeMessage()]}
          streamingContent="The current price of"
          isStreaming={true}
        />,
      );

      expect(screen.getByText('The current price of')).toBeInTheDocument();
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });

    it('does not show empty state when streaming with no messages', () => {
      render(
        <MessageList
          messages={[]}
          streamingContent=""
          isStreaming={true}
        />,
      );

      // When isStreaming is true the component skips the empty state branch
      expect(screen.queryByText('What would you like to know?')).not.toBeInTheDocument();
    });
  });
});

// ---------------------------------------------------------------------------
// ChatInput
// ---------------------------------------------------------------------------
describe('ChatInput', () => {
  it('renders the textarea and send button', () => {
    render(<ChatInput onSend={vi.fn()} />);

    expect(
      screen.getByPlaceholderText(/Ask about stocks/),
    ).toBeInTheDocument();
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('calls onSend with trimmed text when send button is clicked', async () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);

    const textarea = screen.getByPlaceholderText(/Ask about stocks/);
    await userEvent.type(textarea, '  Buy AAPL  ');
    await userEvent.click(screen.getByRole('button'));

    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith('Buy AAPL');
  });

  it('clears the textarea after sending', async () => {
    render(<ChatInput onSend={vi.fn()} />);

    const textarea = screen.getByPlaceholderText(/Ask about stocks/) as HTMLTextAreaElement;
    await userEvent.type(textarea, 'Hello');
    await userEvent.click(screen.getByRole('button'));

    expect(textarea.value).toBe('');
  });

  it('submits on Enter key press', async () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);

    const textarea = screen.getByPlaceholderText(/Ask about stocks/);
    await userEvent.type(textarea, 'Check TSLA{Enter}');

    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith('Check TSLA');
  });

  it('does not submit on Shift+Enter (allows multiline)', async () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);

    const textarea = screen.getByPlaceholderText(/Ask about stocks/);
    await userEvent.type(textarea, 'Line 1{Shift>}{Enter}{/Shift}Line 2');

    expect(onSend).not.toHaveBeenCalled();
  });

  it('does not submit when textarea is empty or whitespace only', async () => {
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);

    // Click send with empty input
    await userEvent.click(screen.getByRole('button'));
    expect(onSend).not.toHaveBeenCalled();

    // Type only spaces and press Enter
    const textarea = screen.getByPlaceholderText(/Ask about stocks/);
    await userEvent.type(textarea, '   {Enter}');
    expect(onSend).not.toHaveBeenCalled();
  });

  it('disables textarea and button when disabled prop is true', () => {
    render(<ChatInput onSend={vi.fn()} disabled />);

    const textarea = screen.getByPlaceholderText(/Ask about stocks/);
    const button = screen.getByRole('button');

    expect(textarea).toBeDisabled();
    expect(button).toBeDisabled();
  });

  it('does not call onSend when disabled even if there is text', async () => {
    const onSend = vi.fn();
    const { rerender } = render(<ChatInput onSend={onSend} />);

    const textarea = screen.getByPlaceholderText(/Ask about stocks/);
    await userEvent.type(textarea, 'Buy AAPL');

    // Now disable and try to submit
    rerender(<ChatInput onSend={onSend} disabled />);
    await userEvent.click(screen.getByRole('button'));

    expect(onSend).not.toHaveBeenCalled();
  });

  it('disables send button when textarea is empty', () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.getByRole('button')).toBeDisabled();
  });
});
