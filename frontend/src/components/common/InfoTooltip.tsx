import { QuestionMarkCircleIcon } from '@heroicons/react/20/solid';
import Tooltip from './Tooltip';

interface InfoTooltipProps {
  text: string;
  position?: 'top' | 'bottom';
}

export default function InfoTooltip({ text, position = 'top' }: InfoTooltipProps) {
  return (
    <Tooltip content={text} position={position}>
      <button
        type="button"
        tabIndex={0}
        className="inline-flex cursor-help text-gray-400 hover:text-gray-500"
        aria-label="More info"
      >
        <QuestionMarkCircleIcon className="h-4 w-4" />
      </button>
    </Tooltip>
  );
}
