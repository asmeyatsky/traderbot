import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { CheckCircleIcon } from '@heroicons/react/24/outline';
import { useLogin } from '../hooks/use-auth';
import { useOnboardingStore } from '../stores/onboarding-store';

const highlights = [
  'AI-powered trading signals and predictions',
  'Real-time market data and analytics',
  'Automated risk management and controls',
];

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  const { mutate, isPending, error } = useLogin();
  const hasCompletedOnboarding = useOnboardingStore((s) => s.hasCompletedOnboarding);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    mutate(
      { email, password },
      { onSuccess: () => navigate(hasCompletedOnboarding ? '/dashboard' : '/onboarding') },
    );
  }

  return (
    <div className="flex min-h-screen">
      {/* Left branding panel */}
      <div className="hidden flex-col justify-center bg-gradient-to-br from-indigo-950 via-indigo-900 to-slate-900 px-12 md:flex md:w-1/2">
        <span className="text-3xl font-bold text-white">TraderBot</span>
        <p className="mt-4 text-lg text-indigo-200">
          Trade smarter with AI-powered intelligence and automated risk management.
        </p>
        <ul className="mt-8 space-y-4">
          {highlights.map((h) => (
            <li key={h} className="flex items-start gap-3 text-sm text-indigo-100">
              <CheckCircleIcon className="mt-0.5 h-5 w-5 shrink-0 text-indigo-400" />
              {h}
            </li>
          ))}
        </ul>
      </div>

      {/* Right form panel */}
      <div className="flex w-full items-center justify-center px-6 py-12 md:w-1/2">
        <div className="w-full max-w-sm space-y-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-indigo-600 md:hidden">TraderBot</h1>
            <h2 className="mt-2 text-2xl font-bold text-gray-900 md:mt-0">Welcome back</h2>
            <p className="mt-1 text-sm text-gray-600">Sign in to your account</p>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">
                Invalid email or password
              </p>
            )}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
            <button
              type="submit"
              disabled={isPending}
              className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
            >
              {isPending ? 'Signing in...' : 'Sign in'}
            </button>
          </form>
          <p className="text-center text-sm text-gray-600">
            Don't have an account?{' '}
            <Link to="/register" className="font-medium text-indigo-600 hover:text-indigo-500">
              Register
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
