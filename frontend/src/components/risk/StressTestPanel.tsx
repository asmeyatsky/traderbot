import type { StressResult } from '../../types/risk';
import { formatPercent } from '../../lib/format';

export default function StressTestPanel({ results }: { results: StressResult[] }) {
  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h3 className="text-sm font-medium text-gray-500">Stress Test Scenarios</h3>
      <ul className="mt-4 divide-y divide-gray-100">
        {results.map((r) => (
          <li key={r.scenario} className="flex items-center justify-between py-3">
            <div>
              <p className="text-sm font-medium text-gray-900">{r.scenario}</p>
              <p className="text-xs text-gray-500">{r.description}</p>
            </div>
            <div className="text-right">
              <p className={`text-sm font-medium ${r.portfolio_impact < 0 ? 'text-red-600' : 'text-green-600'}`}>
                {formatPercent(r.portfolio_impact * 100)}
              </p>
              <p className="text-xs text-gray-500">P: {(r.probability * 100).toFixed(0)}%</p>
            </div>
          </li>
        ))}
        {results.length === 0 && <li className="py-3 text-sm text-gray-400">No stress test results</li>}
      </ul>
    </div>
  );
}
