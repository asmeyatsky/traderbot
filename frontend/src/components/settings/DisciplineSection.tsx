import { useEffect, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { ShieldCheckIcon, TrashIcon } from '@heroicons/react/24/outline';
import { updateDisciplineRules } from '../../api/auth';
import { useMe } from '../../hooks/use-auth';
import LoadingSpinner from '../common/LoadingSpinner';
import type { User } from '../../types/user';

const MAX_RULES = 50;
const MAX_RULE_LENGTH = 500;
const MAX_PHILOSOPHY_LENGTH = 4000;

/**
 * Phase 10.1 — Discipline Coach settings.
 *
 * Users edit their free-form discipline rules and trading philosophy here.
 * The pre-trade AI veto layer checks every order against every rule; setting
 * no rules + no philosophy keeps the check a no-op (no API call, no cost).
 */
export default function DisciplineSection() {
  const { data: user, isLoading } = useMe();
  const queryClient = useQueryClient();

  const [rules, setRules] = useState<string[]>([]);
  const [ruleInput, setRuleInput] = useState('');
  const [philosophy, setPhilosophy] = useState('');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    if (user) {
      setRules(user.discipline_rules ?? []);
      setPhilosophy(user.trading_philosophy ?? '');
      setHasChanges(false);
    }
  }, [user]);

  const { mutate: save, isPending, isSuccess, error } = useMutation({
    mutationFn: updateDisciplineRules,
    onSuccess: (updatedUser: User) => {
      // Write the new user back into the me cache so sibling sections see it.
      queryClient.setQueryData(['me'], updatedUser);
      setHasChanges(false);
    },
  });

  if (isLoading) return <LoadingSpinner className="py-10" />;

  function addRule() {
    const trimmed = ruleInput.trim();
    if (!trimmed) return;
    if (rules.length >= MAX_RULES) return;
    if (trimmed.length > MAX_RULE_LENGTH) return;
    setRules((r) => [...r, trimmed]);
    setRuleInput('');
    setHasChanges(true);
  }

  function removeRule(idx: number) {
    setRules((r) => r.filter((_, i) => i !== idx));
    setHasChanges(true);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') {
      e.preventDefault();
      addRule();
    }
  }

  function handleSave() {
    save({
      discipline_rules: rules,
      trading_philosophy: philosophy,
    });
  }

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5 dark:bg-gray-800 dark:ring-white/10">
      <div className="flex items-center gap-2">
        <ShieldCheckIcon className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Discipline Coach
        </h2>
      </div>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        Write your own trading rules in plain English. Before any order is
        placed, Claude checks the trade against every rule and can refuse it
        with a specific reason. You can always override a refusal — every
        override is logged.
      </p>

      {isSuccess && (
        <p className="mt-3 rounded-md bg-green-50 p-3 text-sm text-green-700 dark:bg-green-900/30 dark:text-green-400">
          Discipline settings saved
        </p>
      )}
      {error && (
        <p className="mt-3 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-400">
          Couldn&apos;t save — {error instanceof Error ? error.message : 'please try again'}
        </p>
      )}

      {/* Rules list */}
      <div className="mt-4">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Your Rules
          </label>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {rules.length}/{MAX_RULES}
          </span>
        </div>

        {rules.length === 0 ? (
          <p className="mt-2 rounded-md border border-dashed border-gray-300 px-3 py-3 text-sm text-gray-500 dark:border-gray-600 dark:text-gray-400">
            No rules yet. Examples: &ldquo;Never buy meme stocks&rdquo;,
            &ldquo;No more than two new positions per day&rdquo;, &ldquo;Never
            average down on a losing position&rdquo;.
          </p>
        ) : (
          <ul className="mt-2 space-y-2">
            {rules.map((rule, idx) => (
              <li
                key={`${idx}-${rule}`}
                className="group flex items-start gap-3 rounded-md border border-gray-200 bg-gray-50 px-3 py-2 dark:border-gray-600 dark:bg-gray-700/30"
              >
                <span className="mt-0.5 text-xs font-semibold text-gray-500 dark:text-gray-400">
                  #{idx + 1}
                </span>
                <span className="flex-1 text-sm text-gray-800 dark:text-gray-200">
                  {rule}
                </span>
                <button
                  type="button"
                  aria-label={`Remove rule ${idx + 1}`}
                  onClick={() => removeRule(idx)}
                  className="text-gray-400 opacity-0 transition-opacity hover:text-red-600 group-hover:opacity-100 dark:hover:text-red-400"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-3 flex gap-2">
          <input
            value={ruleInput}
            onChange={(e) => setRuleInput(e.target.value)}
            onKeyDown={handleKeyDown}
            maxLength={MAX_RULE_LENGTH}
            placeholder="Add a rule, e.g. 'Never hold leveraged ETFs overnight'"
            className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
            disabled={rules.length >= MAX_RULES}
          />
          <button
            type="button"
            onClick={addRule}
            disabled={!ruleInput.trim() || rules.length >= MAX_RULES}
            className="rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
          >
            Add
          </button>
        </div>
      </div>

      {/* Philosophy */}
      <div className="mt-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Trading Philosophy
          <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
            (optional — used for ambiguous orders where no narrow rule fires)
          </span>
        </label>
        <textarea
          value={philosophy}
          onChange={(e) => {
            setPhilosophy(e.target.value);
            setHasChanges(true);
          }}
          maxLength={MAX_PHILOSOPHY_LENGTH}
          rows={4}
          placeholder="e.g. I'm a long-term dividend investor. I avoid speculation. I prefer companies with 10+ years of earnings growth and never put more than 5% of my portfolio in one name."
          className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400"
        />
        <div className="mt-1 flex justify-end text-xs text-gray-500 dark:text-gray-400">
          {philosophy.length}/{MAX_PHILOSOPHY_LENGTH}
        </div>
      </div>

      <button
        type="button"
        onClick={handleSave}
        disabled={isPending || !hasChanges}
        className="mt-4 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
      >
        {isPending ? 'Saving...' : 'Save Discipline Settings'}
      </button>
    </div>
  );
}
