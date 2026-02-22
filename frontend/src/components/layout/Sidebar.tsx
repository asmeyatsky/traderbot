import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  ChartBarIcon,
  BanknotesIcon,
  ArrowTrendingUpIcon,
  ChartBarSquareIcon,
  ClockIcon,
  Cog6ToothIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', to: '/dashboard', icon: HomeIcon },
  { name: 'Trading', to: '/trading', icon: ArrowTrendingUpIcon },
  { name: 'Portfolio', to: '/portfolio', icon: BanknotesIcon },
  { name: 'Market Data', to: '/market-data', icon: ChartBarSquareIcon },
  { name: 'ML Predictions', to: '/predictions', icon: CpuChipIcon },
  { name: 'Risk Analytics', to: '/analytics', icon: ChartBarIcon },
  { name: 'Activity', to: '/activity', icon: ClockIcon },
  { name: 'Settings', to: '/settings', icon: Cog6ToothIcon },
];

export default function Sidebar() {
  return (
    <aside className="flex w-64 flex-col border-r border-gray-200 bg-white">
      <div className="flex h-16 items-center px-6">
        <span className="text-xl font-bold text-indigo-600">TraderBot</span>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-indigo-50 text-indigo-600'
                  : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
              }`
            }
          >
            <item.icon className="h-5 w-5" />
            {item.name}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
