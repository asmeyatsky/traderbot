import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { CheckCircleIcon, ShieldCheckIcon, BoltIcon } from '@heroicons/react/24/outline';
import { ScaleIcon } from '@heroicons/react/24/outline';
import { useRegister, useLogin } from '../hooks/use-auth';
import StepIndicator from '../components/common/StepIndicator';

const riskOptions = [
  { value: 'CONSERVATIVE', label: 'Conservative', description: 'Steady growth, lower volatility', icon: ShieldCheckIcon },
  { value: 'MODERATE', label: 'Moderate', description: 'Balanced risk and return', icon: ScaleIcon },
  { value: 'AGGRESSIVE', label: 'Aggressive', description: 'Maximum growth potential', icon: BoltIcon },
];

const goalOptions = [
  { value: 'CAPITAL_PRESERVATION', label: 'Capital Preservation', description: 'Protect your capital above all' },
  { value: 'BALANCED_GROWTH', label: 'Balanced Growth', description: 'Grow wealth with managed risk' },
  { value: 'MAXIMUM_RETURNS', label: 'Maximum Returns', description: 'Pursue highest possible returns' },
];

const stepLabels = ['Account', 'Risk Profile', 'Goal'];

const highlights = [
  'AI-powered trading signals and predictions',
  'Real-time market data and analytics',
  'Automated risk management and controls',
];

export default function RegisterPage() {
  const [step, setStep] = useState(0);
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

  function update(field: string, value: string) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function handleNext() {
    if (step < 2) {
      setStep(step + 1);
    } else {
      registerMutation.mutate(form, {
        onSuccess: () => {
          loginMutation.mutate(
            { email: form.email, password: form.password },
            { onSuccess: () => navigate('/onboarding') },
          );
        },
      });
    }
  }

  function canAdvance() {
    if (step === 0) return form.first_name && form.last_name && form.email && form.password.length >= 8;
    return true;
  }

  return (
    <div className="flex min-h-screen">
      {/* Left branding panel */}
      <div className="hidden flex-col justify-center bg-gradient-to-br from-indigo-950 via-indigo-900 to-slate-900 px-12 md:flex md:w-1/2">
        <span className="text-3xl font-bold text-white">TraderBot</span>
        <p className="mt-4 text-lg text-indigo-200">
          Create your account and start trading with AI-powered insights.
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
        <div className="w-full max-w-md space-y-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-indigo-600 md:hidden">TraderBot</h1>
            <h2 className="mt-2 text-2xl font-bold text-gray-900 md:mt-0">Create your account</h2>
          </div>

          <StepIndicator steps={stepLabels} currentStep={step} />

          {error && (
            <p className="rounded-md bg-red-50 p-3 text-sm text-red-700">
              {(error as Error).message || 'Registration failed'}
            </p>
          )}

          {/* Step 0: Account details */}
          {step === 0 && (
            <div className="space-y-4">
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
                <p className="mt-1 text-xs text-gray-400">At least 8 characters</p>
              </div>
            </div>
          )}

          {/* Step 1: Risk Profile */}
          {step === 1 && (
            <div className="space-y-4">
              <p className="text-center text-sm text-gray-600">Select your risk tolerance</p>
              <div className="grid grid-cols-1 gap-3">
                {riskOptions.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => update('risk_tolerance', opt.value)}
                    className={`flex items-center gap-4 rounded-lg border-2 p-4 text-left transition-colors ${
                      form.risk_tolerance === opt.value
                        ? 'border-indigo-600 bg-indigo-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <opt.icon className={`h-8 w-8 shrink-0 ${form.risk_tolerance === opt.value ? 'text-indigo-600' : 'text-gray-400'}`} />
                    <div>
                      <div className="text-sm font-semibold text-gray-900">{opt.label}</div>
                      <div className="text-xs text-gray-500">{opt.description}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 2: Investment Goal */}
          {step === 2 && (
            <div className="space-y-4">
              <p className="text-center text-sm text-gray-600">Choose your investment goal</p>
              <div className="grid grid-cols-1 gap-3">
                {goalOptions.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => update('investment_goal', opt.value)}
                    className={`rounded-lg border-2 p-4 text-left transition-colors ${
                      form.investment_goal === opt.value
                        ? 'border-indigo-600 bg-indigo-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-sm font-semibold text-gray-900">{opt.label}</div>
                    <div className="text-xs text-gray-500">{opt.description}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex items-center justify-between">
            {step > 0 ? (
              <button
                type="button"
                onClick={() => setStep(step - 1)}
                className="text-sm font-medium text-gray-600 hover:text-gray-900"
              >
                Back
              </button>
            ) : (
              <span />
            )}
            <button
              type="button"
              onClick={handleNext}
              disabled={isPending || !canAdvance()}
              className="rounded-md bg-indigo-600 px-6 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
            >
              {step < 2 ? 'Next' : isPending ? 'Creating account...' : 'Create account'}
            </button>
          </div>

          <p className="text-center text-sm text-gray-600">
            Already have an account?{' '}
            <Link to="/login" className="font-medium text-indigo-600 hover:text-indigo-500">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
