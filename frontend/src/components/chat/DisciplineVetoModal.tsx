import type { DisciplineVeto } from '../../types/order';

interface Props {
  open: boolean;
  vetoes: DisciplineVeto[];
  isOverriding: boolean;
  onOverride: () => void;
  onCancel: () => void;
}

/**
 * Phase 10.1 — "break this rule?" modal.
 *
 * Shown when the backend refuses an order with HTTP 400 + body
 * `{error: 'discipline_veto', vetoes: [...]}`. The modal surfaces every
 * violated rule (user's own wording) with the AI's one-sentence evidence.
 *
 * The "Override and proceed" button re-submits the identical order with
 * `override_discipline_vetoes: true` — every use is audited server-side
 * via an OrderVetoOverrideRequested event. The user must click a second
 * confirm button before the override fires, so accidental double-clicks
 * don't silently bypass their own rules.
 */
export default function DisciplineVetoModal({
  open,
  vetoes,
  isOverriding,
  onOverride,
  onCancel,
}: Props) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="discipline-veto-title"
    >
      <div className="w-full max-w-md rounded-lg bg-white shadow-xl dark:bg-gray-800">
        <div className="p-6">
          <h2
            id="discipline-veto-title"
            className="text-lg font-semibold text-gray-900 dark:text-white"
          >
            This order breaks your rules
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Your discipline coach flagged {vetoes.length === 1 ? 'a rule' : `${vetoes.length} rules`} you set for
            yourself. Review before you override.
          </p>

          <ul className="mt-4 space-y-3">
            {vetoes.map((v, idx) => (
              <li
                key={`${v.rule_id}-${idx}`}
                className="rounded-md border border-amber-200 bg-amber-50 p-3 dark:border-amber-900/40 dark:bg-amber-900/20"
              >
                <p className="text-sm font-semibold text-amber-900 dark:text-amber-300">
                  {v.rule_text}
                </p>
                <p className="mt-1 text-xs text-amber-800 dark:text-amber-400">
                  {v.evidence}
                </p>
              </li>
            ))}
          </ul>

          <div className="mt-6 flex flex-col gap-2 sm:flex-row-reverse">
            <button
              type="button"
              onClick={onOverride}
              disabled={isOverriding}
              className="rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-red-700 disabled:opacity-50"
            >
              {isOverriding ? 'Submitting…' : 'Override and proceed'}
            </button>
            <button
              type="button"
              onClick={onCancel}
              className="rounded-md bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
            >
              Cancel
            </button>
          </div>
          <p className="mt-3 text-[11px] text-gray-400 dark:text-gray-500">
            Every override is logged. You can edit or remove these rules in
            Settings → Discipline Coach at any time.
          </p>
        </div>
      </div>
    </div>
  );
}
