import { useState } from 'react';
import { usePrediction, useSignal } from '../hooks/use-ml';
import { useAuthStore } from '../stores/auth-store';
import SymbolSearch from '../components/market-data/SymbolSearch';
import PredictionCard from '../components/ml/PredictionCard';
import SignalGauge from '../components/ml/SignalGauge';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorAlert from '../components/common/ErrorAlert';

export default function MLPredictionsPage() {
  const [symbol, setSymbol] = useState('');
  const userId = useAuthStore((s) => s.user?.id);
  const { data: prediction, isLoading: loadingPred, error: predError } = usePrediction(symbol);
  const { data: signal, isLoading: loadingSig } = useSignal(symbol, userId);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">ML Predictions</h1>
      <SymbolSearch onSearch={setSymbol} currentSymbol={symbol} />

      {!symbol && <p className="py-10 text-center text-gray-500">Enter a symbol to get predictions</p>}

      {(loadingPred || loadingSig) && <LoadingSpinner className="py-10" />}
      {predError && <ErrorAlert message={`Failed to get prediction for ${symbol}`} />}

      {prediction && signal && (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <PredictionCard data={prediction} />
          <SignalGauge data={signal} />
        </div>
      )}
    </div>
  );
}
