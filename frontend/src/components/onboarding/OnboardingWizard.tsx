import { useState } from 'react';
import StepIndicator from '../common/StepIndicator';
import WelcomeStep from './WelcomeStep';
import FundAccountStep from './FundAccountStep';
import ExploreStep from './ExploreStep';
import FirstTradeStep from './FirstTradeStep';
import CompletionStep from './CompletionStep';

const stepLabels = ['Welcome', 'Fund', 'Explore', 'Trade', 'Done'];

export default function OnboardingWizard() {
  const [step, setStep] = useState(0);
  const [tradeSymbol, setTradeSymbol] = useState('AAPL');

  function next() {
    setStep((s) => Math.min(s + 1, 4));
  }

  function handleExploreNext(symbol: string) {
    setTradeSymbol(symbol);
    next();
  }

  return (
    <div className="mx-auto max-w-lg px-6 py-12">
      <div className="mb-10">
        <StepIndicator steps={stepLabels} currentStep={step} />
      </div>

      {step === 0 && <WelcomeStep onNext={next} />}
      {step === 1 && <FundAccountStep onNext={next} />}
      {step === 2 && <ExploreStep onNext={handleExploreNext} />}
      {step === 3 && <FirstTradeStep symbol={tradeSymbol} onNext={next} />}
      {step === 4 && <CompletionStep />}
    </div>
  );
}
