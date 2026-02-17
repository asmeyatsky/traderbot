import {
  SparklesIcon,
  ShieldCheckIcon,
  ChartBarSquareIcon,
  PresentationChartLineIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/auth-store';

const features = [
  { icon: SparklesIcon, title: 'AI Predictions', description: 'ML-powered trading signals' },
  { icon: ShieldCheckIcon, title: 'Risk Controls', description: 'Automated position management' },
  { icon: ChartBarSquareIcon, title: 'Market Data', description: 'Real-time quotes and charts' },
  { icon: PresentationChartLineIcon, title: 'Analytics', description: 'Performance tracking' },
];

interface WelcomeStepProps {
  onNext: () => void;
}

export default function WelcomeStep({ onNext }: WelcomeStepProps) {
  const firstName = useAuthStore((s) => s.user?.first_name ?? 'there');

  return (
    <div className="animate-fade-in text-center">
      <h2 className="text-3xl font-bold text-gray-900">Welcome, {firstName}!</h2>
      <p className="mt-3 text-gray-600">
        Let's get you set up so you can start trading with confidence.
      </p>
      <div className="mt-10 grid grid-cols-2 gap-4">
        {features.map((f) => (
          <div key={f.title} className="rounded-lg bg-gray-50 p-4 text-center">
            <f.icon className="mx-auto h-8 w-8 text-indigo-600" />
            <div className="mt-2 text-sm font-semibold text-gray-900">{f.title}</div>
            <div className="text-xs text-gray-500">{f.description}</div>
          </div>
        ))}
      </div>
      <button
        onClick={onNext}
        className="mt-10 rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700"
      >
        Let's Get Started
      </button>
    </div>
  );
}
