import { useState, useEffect } from 'react';
import { useMe, useUpdateMe } from '../hooks/use-auth';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { RISK_TOLERANCES, INVESTMENT_GOALS } from '../lib/constants';

export default function SettingsPage() {
  const { data: user, isLoading } = useMe();
  const { mutate: update, isPending, isSuccess } = useUpdateMe();
  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    risk_tolerance: '',
    investment_goal: '',
  });

  useEffect(() => {
    if (user) {
      setForm({
        first_name: user.first_name,
        last_name: user.last_name,
        risk_tolerance: user.risk_tolerance ?? 'MODERATE',
        investment_goal: user.investment_goal ?? 'BALANCED_GROWTH',
      });
    }
  }, [user]);

  if (isLoading) return <LoadingSpinner className="py-20" />;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    update(form);
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
      <form onSubmit={handleSubmit} className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
        {isSuccess && (
          <p className="mb-4 rounded-md bg-green-50 p-3 text-sm text-green-700">Settings saved</p>
        )}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">First Name</label>
              <input
                value={form.first_name}
                onChange={(e) => setForm((f) => ({ ...f, first_name: e.target.value }))}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">Last Name</label>
              <input
                value={form.last_name}
                onChange={(e) => setForm((f) => ({ ...f, last_name: e.target.value }))}
                className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Email</label>
            <input
              type="email"
              value={user?.email ?? ''}
              disabled
              className="mt-1 block w-full rounded-md border border-gray-200 bg-gray-50 px-3 py-2 text-gray-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Risk Tolerance</label>
            <select
              value={form.risk_tolerance}
              onChange={(e) => setForm((f) => ({ ...f, risk_tolerance: e.target.value }))}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              {RISK_TOLERANCES.map((r) => (
                <option key={r} value={r}>{r.replace('_', ' ')}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Investment Goal</label>
            <select
              value={form.investment_goal}
              onChange={(e) => setForm((f) => ({ ...f, investment_goal: e.target.value }))}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            >
              {INVESTMENT_GOALS.map((g) => (
                <option key={g} value={g}>{g.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
        </div>
        <button
          type="submit"
          disabled={isPending}
          className="mt-6 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {isPending ? 'Saving...' : 'Save Changes'}
        </button>
      </form>
    </div>
  );
}
