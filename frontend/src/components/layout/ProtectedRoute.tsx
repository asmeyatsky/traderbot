import { useEffect } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../stores/auth-store';
import { useOnboardingStore } from '../../stores/onboarding-store';

export default function ProtectedRoute() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const hasCompletedOnboarding = useOnboardingStore((s) => s.hasCompletedOnboarding);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isAuthenticated) {
      navigate('/login', { replace: true });
    } else if (!hasCompletedOnboarding && location.pathname !== '/onboarding') {
      navigate('/onboarding', { replace: true });
    }
  }, [isAuthenticated, hasCompletedOnboarding, location.pathname, navigate]);

  if (!isAuthenticated) return null;
  if (!hasCompletedOnboarding && location.pathname !== '/onboarding') return null;
  return <Outlet />;
}
