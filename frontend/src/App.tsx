import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './stores/auth-store';
import AppShell from './components/layout/AppShell';
import ProtectedRoute from './components/layout/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import OnboardingPage from './pages/OnboardingPage';
import ChatPage from './pages/ChatPage';
import PortfolioPage from './pages/PortfolioPage';
import MarketsPage from './pages/MarketsPage';
import SettingsPage from './pages/SettingsPage';
import BacktestPage from './pages/BacktestPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 30_000,
    },
  },
});

function RootRedirect() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const navigate = useNavigate();

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/chat', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  return isAuthenticated ? null : <LandingPage />;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<RootRedirect />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/onboarding" element={<OnboardingPage />} />
            <Route element={<AppShell />}>
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/portfolio" element={<PortfolioPage />} />
              <Route path="/markets" element={<MarketsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/backtest" element={<BacktestPage />} />
              {/* Legacy redirects */}
              <Route path="/dashboard" element={<Navigate to="/chat" replace />} />
              <Route path="/trading" element={<Navigate to="/chat" replace />} />
              <Route path="/market-data" element={<Navigate to="/markets" replace />} />
              <Route path="/predictions" element={<Navigate to="/chat" replace />} />
              <Route path="/analytics" element={<Navigate to="/chat" replace />} />
              <Route path="/activity" element={<Navigate to="/chat" replace />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
