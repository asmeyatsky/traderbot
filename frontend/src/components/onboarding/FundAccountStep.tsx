import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { depositCash } from '../../api/portfolio';
import { formatCurrency } from '../../lib/format';

const presets = [1000, 5000, 10000];

interface FundAccountStepProps {
  onNext: () => void;
}

export default function FundAccountStep({ onNext }: FundAccountStepProps) {
  const [amount, setAmount] = useState<number | ''>('');
  const [selectedPreset, setSelectedPreset] = useState<number | null>(null);
  const [success, setSuccess] = useState(false);
  const queryClient = useQueryClient();

  const { mutate, isPending, error } = useMutation({
    mutationFn: depositCash,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      setSuccess(true);
    },
  });

  const effectiveAmount = selectedPreset ?? (typeof amount === 'number' ? amount : 0);

  function handleDeposit() {
    if (effectiveAmount > 0) mutate(effectiveAmount);
  }

  function selectPreset(val: number) {
    setSelectedPreset(val);
    setAmount('');
  }

  function handleCustom(val: string) {
    setSelectedPreset(null);
    setAmount(val === '' ? '' : Number(val));
  }

  if (success) {
    return (
      <div className="animate-fade-in text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
          <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
          </svg>
        </div>
        <h3 className="mt-4 text-lg font-semibold text-gray-900">
          {formatCurrency(effectiveAmount)} deposited!
        </h3>
        <p className="mt-2 text-sm text-gray-500">Your account is funded and ready to go.</p>
        <button
          onClick={onNext}
          className="mt-8 rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700"
        >
          Continue
        </button>
      </div>
    );
  }

  return (
    <div className="animate-fade-in text-center">
      <h2 className="text-2xl font-bold text-gray-900">Fund Your Paper Account</h2>
      <p className="mt-2 text-sm text-gray-600">
        Add virtual cash to start practicing trades risk-free. No real money is involved — you can always add more later.
      </p>

      <div className="mt-8 flex justify-center gap-3">
        {presets.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => selectPreset(p)}
            className={`rounded-lg border-2 px-5 py-3 text-sm font-semibold transition-colors ${
              selectedPreset === p
                ? 'border-indigo-600 bg-indigo-50 text-indigo-700'
                : 'border-gray-200 text-gray-700 hover:border-gray-300'
            }`}
          >
            {formatCurrency(p)}
          </button>
        ))}
      </div>

      <div className="mx-auto mt-4 max-w-xs">
        <input
          type="number"
          min={1}
          placeholder="Custom amount"
          value={amount}
          onChange={(e) => handleCustom(e.target.value)}
          className="block w-full rounded-md border border-gray-300 px-3 py-2 text-center shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
        />
      </div>

      {error && (
        <p className="mt-3 text-sm text-red-600">Failed to deposit. Please try again.</p>
      )}

      <div className="mt-8 flex items-center justify-center gap-4">
        <button
          onClick={onNext}
          className="text-sm font-medium text-gray-500 hover:text-gray-700"
        >
          Skip for Now
        </button>
        <button
          onClick={handleDeposit}
          disabled={isPending || effectiveAmount <= 0}
          className="rounded-md bg-indigo-600 px-8 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
        >
          {isPending ? 'Depositing...' : 'Deposit Funds'}
        </button>
      </div>
    </div>
  );
}
