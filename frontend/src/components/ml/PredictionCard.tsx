import type { Prediction } from '../../types/ml';
import { formatCurrency, formatPercent } from '../../lib/format';
import { ML_HELP } from '../../lib/help-text';
import InfoTooltip from '../common/InfoTooltip';

export default function PredictionCard({ data }: { data: Prediction }) {
  const isUp = data.predicted_direction === 'UP';
  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-gray-900">{data.symbol}</h3>
        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${isUp ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          {data.predicted_direction}
        </span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
        <div>
          <span className="text-gray-500">Current</span>
          <p className="font-medium">{formatCurrency(data.current_price)}</p>
        </div>
        <div>
          <span className="text-gray-500">Predicted</span>
          <p className="font-medium">{formatCurrency(data.predicted_price)}</p>
        </div>
        <div>
          <span className="flex items-center gap-1 text-gray-500">
            Confidence
            <InfoTooltip text={ML_HELP.confidence} />
          </span>
          <p className="font-medium">{formatPercent(data.confidence * 100)}</p>
        </div>
        <div>
          <span className="flex items-center gap-1 text-gray-500">
            Score
            <InfoTooltip text={ML_HELP.score} />
          </span>
          <p className="font-medium">{data.score.toFixed(2)}</p>
        </div>
      </div>
      <p className="mt-3 text-xs text-gray-500">{data.explanation}</p>
    </div>
  );
}
