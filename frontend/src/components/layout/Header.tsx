import { ArrowRightOnRectangleIcon, UserCircleIcon } from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/auth-store';
import { useLogout } from '../../hooks/use-auth';
import Tooltip from '../common/Tooltip';

export default function Header() {
  const user = useAuthStore((s) => s.user);
  const { mutate: doLogout } = useLogout();

  return (
    <header className="flex h-16 items-center justify-between border-b border-gray-200 bg-white px-6">
      <Tooltip content="You are trading with virtual money. No real funds are at risk." position="bottom">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700 ring-1 ring-amber-200">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
          Paper Trading
        </span>
      </Tooltip>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm text-gray-700">
          <UserCircleIcon className="h-6 w-6 text-gray-400" />
          <span>{user ? `${user.first_name} ${user.last_name}` : ''}</span>
        </div>
        <button
          onClick={() => doLogout()}
          className="flex items-center gap-1 rounded-md px-3 py-1.5 text-sm text-gray-500 hover:bg-gray-100 hover:text-gray-700"
        >
          <ArrowRightOnRectangleIcon className="h-4 w-4" />
          Logout
        </button>
      </div>
    </header>
  );
}
