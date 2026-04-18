import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: true,
    css: false,

    // 2026 rules §5: enforce coverage at the layers that carry business logic.
    // UI components are excluded — they're tested via integration/e2e where the
    // value/cost ratio is better. stores/ and api/ are the launch-critical
    // surfaces the AI interacts with.
    //
    // Floor is GLOBAL (not per-file) to start. Only auth-store is tested today;
    // the floor ratchets up as tests are added. Target by Phase 8 launch: 70%.
    // Current baseline is ~9% lines, so 8% is the first stable floor. Raise
    // every time a new store or api module gets tests.
    coverage: {
      provider: 'v8',
      include: ['src/stores/**', 'src/api/**'],
      exclude: ['**/*.test.ts', '**/*.test.tsx', '**/test/**'],
      reporter: ['text', 'text-summary'],
      thresholds: {
        lines: 7,        // baseline 9.18% — TODO(phase-8): raise to 70 before launch
        statements: 7,   // baseline 7.94% — TODO(phase-8): raise to 70
        branches: 2,     // baseline 2.56% — TODO(phase-8): raise to 60
        functions: 7,    // baseline 8.79% — TODO(phase-8): raise to 70
      },
    },
  },
});
