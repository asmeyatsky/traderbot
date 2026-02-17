import { useNavigate } from 'react-router-dom';
import { useOnboardingStore } from '../stores/onboarding-store';
import OnboardingWizard from '../components/onboarding/OnboardingWizard';

export default function OnboardingPage() {
  const navigate = useNavigate();
  const markComplete = useOnboardingStore((s) => s.markComplete);

  function handleSkip() {
    markComplete();
    navigate('/dashboard');
  }

  return (
    <div className="min-h-screen bg-white">
      <header className="flex items-center justify-between px-6 py-4">
        <span className="text-xl font-bold text-indigo-600">TraderBot</span>
        <button
          onClick={handleSkip}
          className="text-sm font-medium text-gray-500 hover:text-gray-700"
        >
          Skip Setup
        </button>
      </header>
      <OnboardingWizard />
    </div>
  );
}
