import { SunIcon, MoonIcon, ComputerDesktopIcon } from '@heroicons/react/24/outline';
import { useThemeStore } from '../../stores/theme-store';

const options = [
  { value: 'light' as const, icon: SunIcon, label: 'Light' },
  { value: 'dark' as const, icon: MoonIcon, label: 'Dark' },
  { value: 'system' as const, icon: ComputerDesktopIcon, label: 'System' },
];

export default function ThemeToggle() {
  const theme = useThemeStore((s) => s.theme);
  const setTheme = useThemeStore((s) => s.setTheme);

  return (
    <div className="flex rounded-lg bg-gray-100 p-0.5 dark:bg-gray-700">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => setTheme(opt.value)}
          className={`flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
            theme === opt.value
              ? 'bg-white text-gray-900 shadow-sm dark:bg-gray-600 dark:text-white'
              : 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
          }`}
          title={opt.label}
        >
          <opt.icon className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">{opt.label}</span>
        </button>
      ))}
    </div>
  );
}
