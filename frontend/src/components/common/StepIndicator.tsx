interface StepIndicatorProps {
  steps: string[];
  currentStep: number;
}

export default function StepIndicator({ steps, currentStep }: StepIndicatorProps) {
  return (
    <div className="flex items-center justify-center gap-3">
      {steps.map((label, i) => (
        <div key={label} className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold ${
                i < currentStep
                  ? 'bg-indigo-600 text-white'
                  : i === currentStep
                    ? 'bg-indigo-600 text-white ring-2 ring-indigo-300 ring-offset-2'
                    : 'bg-gray-200 text-gray-500'
              }`}
            >
              {i < currentStep ? '\u2713' : i + 1}
            </span>
            <span
              className={`hidden text-xs font-medium sm:inline ${
                i <= currentStep ? 'text-indigo-600' : 'text-gray-400'
              }`}
            >
              {label}
            </span>
          </div>
          {i < steps.length - 1 && (
            <div className={`h-px w-8 ${i < currentStep ? 'bg-indigo-600' : 'bg-gray-200'}`} />
          )}
        </div>
      ))}
    </div>
  );
}
