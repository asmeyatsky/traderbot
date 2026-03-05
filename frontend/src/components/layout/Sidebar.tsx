import { NavLink } from 'react-router-dom';
import {
  ChatBubbleLeftRightIcon,
  BanknotesIcon,
  ChartBarSquareIcon,
  Cog6ToothIcon,
  BeakerIcon,
  TrophyIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Chat', to: '/chat', icon: ChatBubbleLeftRightIcon },
  { name: 'Portfolio', to: '/portfolio', icon: BanknotesIcon },
  { name: 'Markets', to: '/markets', icon: ChartBarSquareIcon },
  { name: 'Backtest', to: '/backtest', icon: BeakerIcon },
  { name: 'Leaderboard', to: '/leaderboard', icon: TrophyIcon },
  { name: 'Settings', to: '/settings', icon: Cog6ToothIcon },
];

export default function Sidebar() {
  return (
    <aside className="hidden md:flex w-16 hover:w-48 flex-col border-r border-gray-200 bg-white transition-all duration-200 group/sidebar overflow-hidden dark:border-gray-700 dark:bg-gray-800">
      <div className="flex h-16 items-center justify-center px-3">
        <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">T</span>
        <span className="overflow-hidden whitespace-nowrap text-xl font-bold text-indigo-600 opacity-0 transition-opacity duration-200 group-hover/sidebar:opacity-100 dark:text-indigo-400">
          raderBot
        </span>
      </div>
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400'
                  : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-gray-200'
              }`
            }
          >
            <item.icon className="h-5 w-5 shrink-0" />
            <span className="overflow-hidden whitespace-nowrap opacity-0 transition-opacity duration-200 group-hover/sidebar:opacity-100">
              {item.name}
            </span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
