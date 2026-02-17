import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface OnboardingState {
  hasCompletedOnboarding: boolean;
  markComplete: () => void;
  reset: () => void;
}

export const useOnboardingStore = create<OnboardingState>()(
  persist(
    (set) => ({
      hasCompletedOnboarding: false,
      markComplete: () => set({ hasCompletedOnboarding: true }),
      reset: () => set({ hasCompletedOnboarding: false }),
    }),
    { name: 'traderbot-onboarding' },
  ),
);
