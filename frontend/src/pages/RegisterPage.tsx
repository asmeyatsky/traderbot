import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useRegister, useLogin } from '../hooks/use-auth';
import { RISK_TOLERANCES, INVESTMENT_GOALS } from '../lib/constants';

export default function RegisterPage() {
  const [form, setForm] = useState({
    email: '',
    first_name: '',
    last_name: '',
    password: '',
    risk_tolerance: 'MODERATE',
    investment_goal: 'BALANCED_GROWTH',
  });
  const navigate = useNavigate();
  const registerMutation = useRegister();
  const loginMutation = useLogin();

  const isPending = registerMutation.isPending || loginMutation.isPending;
  const error = registerMutation.error || loginMutation.error;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    registerMutation.mutate(form, {
      onSuccess: () => {
        loginMutation.mutate(
          { email: form.email, password: form.password },
          { onSuccess: () => navigate('/') },
        );
      },
    });
  }

  function update(field: string, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  return (
    <div className="flex min-h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm space-y-8">
        <div className="text-center">
          <h1 className="text-3xl font-bold text-indigo-600">TraderBot</h1>
          <p className="mt-2 text-sm text-gray-600">Create your account</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">
              {(error as Error).message || 'Registration failed'}
            </p>
          )}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="first_name" className="block text-sm font-medium text-gray-700">First Name</label>
              <input
                id="first_name"
                value={form.first_name}
                onChange={(e) => update('first_name', e.target.value)}
                required
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label htmlFor="last_name" className="block text-sm font-medium text-gray-700">Last Name</label>
              <input
                id="last_name"
                value={form.last_name}
                onChange={(e) => update('last_name', e.target.value)}
                required
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
          </div>
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
            <input
              id="email"
              type="email"
              value={form.email}
              onChange={(e) => update('email', e.target.value)}
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">Password</label>
            <input
              id="password"
              type="password"
              value={form.password}
              onChange={(e) => update('password', e.target.value)}
              required
              minLength={8}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label htmlFor="risk_tolerance" className="block text-sm font-medium text-gray-700">Risk Tolerance</label>
            <select
              id="risk_tolerance"
              value={form.risk_tolerance}
              onChange={(e) => update('risk_tolerance', e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              {RISK_TOLERANCES.map((r) => (
                <option key={r} value={r}>{r.replace('_', ' ')}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="investment_goal" className="block text-sm font-medium text-gray-700">Investment Goal</label>
            <select
              id="investment_goal"
              value={form.investment_goal}
              onChange={(e) => update('investment_goal', e.target.value)}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              {INVESTMENT_GOALS.map((g) => (
                <option key={g} value={g}>{g.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            disabled={isPending}
            className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
          >
            {isPending ? 'Creating account...' : 'Create account'}
          </button>
        </form>
        <p className="text-center text-sm text-gray-600">
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-indigo-600 hover:text-indigo-500">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
