import { useState } from 'react';
import {
  useBrokerAccounts,
  useLinkBrokerAccount,
  useUpdateBrokerAccount,
  useDeleteBrokerAccount,
} from '../../hooks/use-broker-accounts';
import LoadingSpinner from '../common/LoadingSpinner';

export default function BrokerLinkForm() {
  const { data: accounts, isLoading } = useBrokerAccounts();
  const { mutate: link, isPending: isLinking, isSuccess: linkSuccess } = useLinkBrokerAccount();
  const { mutate: update } = useUpdateBrokerAccount();
  const { mutate: remove } = useDeleteBrokerAccount();

  const [apiKey, setApiKey] = useState('');
  const [secretKey, setSecretKey] = useState('');
  const [paperTrading, setPaperTrading] = useState(true);
  const [showForm, setShowForm] = useState(false);

  if (isLoading) return <LoadingSpinner className="py-6" />;

  const alpacaAccount = accounts?.find((a) => a.broker_type === 'alpaca');

  function handleLink(e: React.FormEvent) {
    e.preventDefault();
    link(
      {
        broker_type: 'alpaca',
        api_key: apiKey,
        secret_key: secretKey,
        paper_trading: paperTrading,
      },
      {
        onSuccess: () => {
          setApiKey('');
          setSecretKey('');
          setShowForm(false);
        },
      },
    );
  }

  return (
    <div className="rounded-lg bg-white p-6 shadow-sm ring-1 ring-gray-900/5">
      <h2 className="text-lg font-semibold text-gray-900">Broker Account</h2>
      <p className="mt-1 text-xs text-gray-500">
        Link your Alpaca brokerage account to execute real trades. Your API keys are encrypted at
        rest.
      </p>

      {alpacaAccount ? (
        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between rounded-lg border border-gray-200 p-4">
            <div>
              <p className="text-sm font-medium text-gray-900">Alpaca</p>
              <p className="text-xs text-gray-500">
                API Key: {alpacaAccount.api_key_hint}
              </p>
              <p className="mt-0.5 text-xs text-gray-500">
                Linked {new Date(alpacaAccount.created_at).toLocaleDateString()}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                  alpacaAccount.paper_trading
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-green-100 text-green-700'
                }`}
              >
                {alpacaAccount.paper_trading ? 'Paper' : 'Live'}
              </span>
            </div>
          </div>

          {/* Paper / Live toggle */}
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700">Trading Mode</label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => update({ id: alpacaAccount.id, paper_trading: true })}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  alpacaAccount.paper_trading
                    ? 'bg-amber-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Paper
              </button>
              <button
                type="button"
                onClick={() => update({ id: alpacaAccount.id, paper_trading: false })}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  !alpacaAccount.paper_trading
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Live
              </button>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setShowForm(true)}
              className="text-sm font-medium text-indigo-600 hover:text-indigo-500"
            >
              Update Keys
            </button>
            <button
              type="button"
              onClick={() => {
                if (confirm('Unlink your Alpaca account? This removes stored credentials.')) {
                  remove(alpacaAccount.id);
                }
              }}
              className="text-sm font-medium text-red-600 hover:text-red-500"
            >
              Unlink
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setShowForm(true)}
          className="mt-4 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700"
        >
          Link Alpaca Account
        </button>
      )}

      {showForm && (
        <form onSubmit={handleLink} className="mt-4 space-y-3 border-t border-gray-200 pt-4">
          {linkSuccess && (
            <p className="rounded-md bg-green-50 p-3 text-sm text-green-700">
              Broker account linked successfully
            </p>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700">API Key</label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="PK..."
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Secret Key</label>
            <input
              type="password"
              value={secretKey}
              onChange={(e) => setSecretKey(e.target.value)}
              placeholder="Enter your secret key"
              required
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="paper-trading"
              checked={paperTrading}
              onChange={(e) => setPaperTrading(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
            />
            <label htmlFor="paper-trading" className="text-sm text-gray-700">
              Paper trading (recommended for testing)
            </label>
          </div>
          <div className="flex gap-2">
            <button
              type="submit"
              disabled={isLinking}
              className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
            >
              {isLinking ? 'Linking...' : 'Link Account'}
            </button>
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="rounded-md bg-gray-100 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200"
            >
              Cancel
            </button>
          </div>
          <p className="text-xs text-gray-400">
            Get your API keys from{' '}
            <a
              href="https://app.alpaca.markets/paper/dashboard/overview"
              target="_blank"
              rel="noopener noreferrer"
              className="text-indigo-600 hover:text-indigo-500"
            >
              Alpaca Dashboard
            </a>
            . Keys are encrypted with AES-256 and never exposed in API responses.
          </p>
        </form>
      )}
    </div>
  );
}
