import type { StressResult } from '../../types/risk';
import { formatPercent } from '../../lib/format';
import { STRESS_TEST_HELP } from '../../lib/help-text';
import InfoTooltip from '../common/InfoTooltip';

export default function StressTestPanel({ results }: { results: StressResult[] }) {
  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-sm font-medium text-gray-500">Stress Test Scenarios</h3>
      <p className="mt-1 text-xs text-gray-400">See how your portfolio might perform under extreme market conditions.</p>
      <ul className="mt-4 divide-y divide-gray-100">
        {results.map((r) => (
          <li key={r.scenario} className="flex items-center justify-between py-3">
            <div>
              <p className="text-sm font-medium text-gray-900">{r.scenario}</p>
              <p className="text-xs text-gray-500">{r.description}</p>
            </div>
            <div className="text-right">
              <p className={`flex items-center gap-1 text-sm font-medium ${r.portfolio_impact < 0 ? 'text-red-600' : 'text-green-600'}`}>
                {formatPercent(r.portfolio_impact * 100)}
                <InfoTooltip text={STRESS_TEST_HELP.impact} />
              </p>
              <p className="flex items-center gap-1 text-xs text-gray-500">
                P: {(r.probability * 100).toFixed(0)}%
                <InfoTooltip text={STRESS_TEST_HELP.probability} />
              </p>
            </div>
          </li>
        ))}
        {results.length === 0 && <li className="py-3 text-sm text-gray-400">No stress test results</li>}
      </ul>
    </div>
  );
}
