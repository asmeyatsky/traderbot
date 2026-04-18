import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import boundaries from 'eslint-plugin-boundaries'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

// Frontend layer rules — mirrors the backend import-linter contracts.
// 2026 rules §2: layered architecture, enforced mechanically in CI.
//
// Layer dependency graph (arrows = "may import from"):
//   pages      → components, hooks, stores, api, lib, types
//   components → hooks, stores, api, lib, types
//   hooks      → stores, api, lib, types
//   stores     → api, lib, types
//   api        → lib, types
//   lib, types → (leaf)
const boundariesElements = [
  { type: 'pages',      pattern: 'pages/*',      mode: 'folder' },
  { type: 'components', pattern: 'components/*', mode: 'folder' },
  { type: 'hooks',      pattern: 'hooks',        mode: 'folder' },
  { type: 'stores',     pattern: 'stores',       mode: 'folder' },
  { type: 'api',        pattern: 'api',          mode: 'folder' },
  { type: 'lib',        pattern: 'lib',          mode: 'folder' },
  { type: 'types',      pattern: 'types',        mode: 'folder' },
  { type: 'bootstrap',  pattern: '{App,main}.{ts,tsx}', mode: 'file' },
  { type: 'test',       pattern: 'test',         mode: 'folder' },
  { type: 'assets',     pattern: 'assets',       mode: 'folder' },
]

// api → stores is intentionally allowed: api clients read the auth token via
// `useAuthStore.getState()` — the idiomatic Zustand read-from-store pattern.
// The stricter boundary that matters (api → components / pages) stays blocked.
const boundariesRules = [
  { from: 'pages',      allow: ['pages', 'components', 'hooks', 'stores', 'api', 'lib', 'types'] },
  { from: 'components', allow: ['components', 'hooks', 'stores', 'api', 'lib', 'types', 'assets'] },
  { from: 'hooks',      allow: ['hooks', 'stores', 'api', 'lib', 'types'] },
  { from: 'stores',     allow: ['stores', 'api', 'lib', 'types'] },
  { from: 'api',        allow: ['api', 'stores', 'lib', 'types'] },
  { from: 'lib',        allow: ['lib', 'types'] },
  { from: 'types',      allow: ['types'] },
  { from: 'bootstrap',  allow: ['bootstrap', 'pages', 'components', 'hooks', 'stores', 'api', 'lib', 'types'] },
  { from: 'test',       allow: ['pages', 'components', 'hooks', 'stores', 'api', 'lib', 'types', 'test'] },
]

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    plugins: {
      boundaries,
    },
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    settings: {
      'boundaries/elements': boundariesElements,
      'boundaries/include': ['src/**/*.{ts,tsx}'],
      'import/resolver': {
        typescript: {
          alwaysTryTypes: true,
          project: './tsconfig.json',
        },
      },
    },
    rules: {
      // Downgrade to warn: common React pattern for syncing server data to form state
      'react-hooks/set-state-in-effect': 'warn',
      'boundaries/dependencies': [
        'error',
        {
          default: 'disallow',
          rules: boundariesRules.map(({ from, allow }) => ({
            from: { type: from },
            allow: { to: { type: allow } },
          })),
        },
      ],
    },
  },
])
