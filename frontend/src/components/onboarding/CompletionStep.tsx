import { Link } from 'react-router-dom';
import {
  HomeIcon,
  ArrowTrendingUpIcon,
  ChartBarSquareIcon,
} from '@heroicons/react/24/outline';
import { useOnboardingStore } from '../../stores/onboarding-store';

const links = [
  { icon: HomeIcon, title: 'Dashboard', description: 'View your portfolio overview', to: '/dashboard' },
  { icon: ArrowTrendingUpIcon, title: 'Trading', description: 'Place orders and manage trades', to: '/trading' },
  { icon: ChartBarSquareIcon, title: 'Market Data', description: 'Explore charts and indicators', to: '/market-data' },
];

export default function CompletionStep() {
  const markComplete = useOnboardingStore((s) => s.markComplete);

  return (
    <div className="animate-fade-in text-center">
      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-indigo-100">
        <svg className="h-8 w-8 text-indigo-600" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
        </svg>
      </div>
      <h2 className="mt-6 text-3xl font-bold text-gray-900">You're All Set!</h2>
      <p className="mt-3 text-gray-600">
        Your account is ready. Explore the platform and start trading with AI-powered insights.
      </p>

      <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-3">
        {links.map((l) => (
          <Link
            key={l.to}
            to={l.to}
            onClick={markComplete}
            className="rounded-lg border border-gray-200 p-4 text-center transition-colors hover:border-indigo-300 hover:bg-indigo-50"
          >
            <l.icon className="mx-auto h-8 w-8 text-indigo-600" />
            <div className="mt-2 text-sm font-semibold text-gray-900">{l.title}</div>
            <div className="text-xs text-gray-500">{l.description}</div>
          </Link>
        ))}
      </div>

      <Link
        to="/dashboard"
        onClick={markComplete}
        className="mt-10 inline-block rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700"
      >
        Go to Dashboard
      </Link>
    </div>
  );
}
