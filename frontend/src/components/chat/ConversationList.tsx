import { PlusIcon, TrashIcon } from '@heroicons/react/24/outline';
import type { Conversation } from '../../types/chat';

interface ConversationListProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export default function ConversationList({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
}: ConversationListProps) {
  return (
    <div className="flex h-full flex-col border-r border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-800/50">
      <div className="flex items-center justify-between p-3">
        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Chats</h2>
        <button
          onClick={onNew}
          className="rounded-lg p-1.5 text-gray-500 transition-colors hover:bg-gray-200 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200"
        >
          <PlusIcon className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {conversations.map((conv) => (
          <button
            key={conv.id}
            onClick={() => onSelect(conv.id)}
            className={`group flex w-full items-center justify-between px-3 py-2.5 text-left text-sm transition-colors ${
              conv.id === activeId
                ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400'
                : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700'
            }`}
          >
            <span className="truncate">{conv.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(conv.id);
              }}
              className="hidden rounded p-0.5 text-gray-400 hover:text-red-500 group-hover:block dark:text-gray-500 dark:hover:text-red-400"
            >
              <TrashIcon className="h-3.5 w-3.5" />
            </button>
          </button>
        ))}

        {conversations.length === 0 && (
          <p className="px-3 py-6 text-center text-xs text-gray-400 dark:text-gray-500">
            No conversations yet
          </p>
        )}
      </div>
    </div>
  );
}
