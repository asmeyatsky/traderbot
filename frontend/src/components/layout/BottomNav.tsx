import { NavLink } from 'react-router-dom';
import {
  ChatBubbleLeftRightIcon,
  BanknotesIcon,
  ChartBarSquareIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';

const items = [
  { name: 'Chat', to: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Portfolio', to: '/portfolio', icon: BanknotesIcon },
  { name: 'Markets', to: '/markets', icon: ChartBarSquareIcon },
  { name: 'Settings', to: '/settings', icon: Cog6ToothIcon },
];

export default function BottomNav() {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-gray-200 bg-white pb-[env(safe-area-inset-bottom)] md:hidden">
      <div className="flex items-center justify-around">
        {items.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) =>
              `flex flex-1 flex-col items-center gap-0.5 py-2 text-[10px] font-medium transition-colors ${
                isActive
                  ? 'text-indigo-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`
            }
          >
            <item.icon className="h-5 w-5" />
            {item.name}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
