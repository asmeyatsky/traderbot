import { useEffect, useState } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import {
  REQUIRED_RISK_PHRASE,
  disableLiveMode,
  enableLiveMode,
  enrollTotp,
  getLiveModeStatus,
  type LiveModeStatus,
  type TotpEnrollmentResponse,
} from '../api/live-mode';

/**
 * Live-trading enablement page (ADR-002).
 *
 * Three steps:
 *   1. KYC attestation — user confirms identity + trading-own-money.
 *   2. TOTP enrollment — generate secret, display QR, user scans in Authy/1Password/etc.
 *   3. Final confirmation — daily cap, exact risk phrase, TOTP code.
 *
 * Every step gates forward progress. Backend re-validates every field so the
 * UI is defence-in-depth rather than the source of truth.
 */
type Step = 'status' | 'kyc' | 'totp' | 'confirm' | 'success';

export default function LiveModePage() {
  const [status, setStatus] = useState<LiveModeStatus | null>(null);
  const [step, setStep] = useState<Step>('status');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // KYC form state
  const [fullName, setFullName] = useState('');
  const [residency, setResidency] = useState('');
  const [over18, setOver18] = useState(false);
  const [ownFunds, setOwnFunds] = useState(false);

  // TOTP enrollment state
  const [totpEnrollment, setTotpEnrollment] = useState<TotpEnrollmentResponse | null>(null);

  // Final form state
  const [dailyCap, setDailyCap] = useState(100);
  const [riskPhrase, setRiskPhrase] = useState('');
  const [totpCode, setTotpCode] = useState('');

  useEffect(() => {
    getLiveModeStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  const kycPayload = JSON.stringify({
    full_name: fullName.trim(),
    residency: residency.trim(),
    over_18: over18,
    own_funds: ownFunds,
    timestamp: new Date().toISOString(),
  });

  const kycReady =
    fullName.trim().length >= 2 &&
    residency.trim().length >= 2 &&
    over18 &&
    ownFunds;

  const confirmReady =
    riskPhrase === REQUIRED_RISK_PHRASE &&
    /^\d{6}$/.test(totpCode) &&
    dailyCap > 0 &&
    dailyCap <= 1000;

  async function handleStartEnrollment() {
    setError(null);
    setBusy(true);
    try {
      const enrollment = await enrollTotp();
      setTotpEnrollment(enrollment);
      setStep('totp');
    } catch (e: unknown) {
      setError(extractError(e, 'Failed to generate TOTP secret'));
    } finally {
      setBusy(false);
    }
  }

  async function handleEnable() {
    setError(null);
    setBusy(true);
    try {
      const s = await enableLiveMode({
        kyc_attestation_payload: kycPayload,
        daily_loss_cap_usd: dailyCap,
        totp_code: totpCode,
        risk_acknowledgement: riskPhrase,
      });
      setStatus(s);
      setStep('success');
    } catch (e: unknown) {
      setError(extractError(e, 'Failed to enable live mode'));
    } finally {
      setBusy(false);
    }
  }

  async function handleDisable() {
    setBusy(true);
    try {
      const s = await disableLiveMode();
      setStatus(s);
      setStep('status');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-6 space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Live Trading</h1>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Switch from simulated trades to real orders on your Alpaca account.
          You can flip back to paper at any time.
        </p>
      </header>

      {error && (
        <div className="rounded border border-red-400 bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-300 px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {step === 'status' && (
        <StatusPanel status={status} onEnable={() => setStep('kyc')} onDisable={handleDisable} busy={busy} />
      )}

      {step === 'kyc' && (
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Step 1 — Identity attestation</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Confirm the details below. These are stored as a cryptographic hash, not plain text.
          </p>
          <LabeledInput label="Full legal name" value={fullName} onChange={setFullName} placeholder="Jane Doe" />
          <LabeledInput label="Country of residence" value={residency} onChange={setResidency} placeholder="United Kingdom" />
          <Checkbox label="I am at least 18 years old." checked={over18} onChange={setOver18} />
          <Checkbox label="I am trading my own money, not on behalf of anyone else." checked={ownFunds} onChange={setOwnFunds} />
          <div className="flex gap-2">
            <button
              className="btn-primary"
              disabled={!kycReady || busy}
              onClick={handleStartEnrollment}
            >
              Continue →
            </button>
            <button className="btn-secondary" onClick={() => setStep('status')}>Cancel</button>
          </div>
        </section>
      )}

      {step === 'totp' && totpEnrollment && (
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Step 2 — Set up 2FA</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Scan this QR code with an authenticator app (Authy, 1Password, Google Authenticator). You'll be asked for a 6-digit code on the next step — and for every live order.
          </p>
          <div className="bg-white p-4 rounded inline-block">
            <QRCodeSVG value={totpEnrollment.provisioning_uri} size={192} />
          </div>
          <details className="text-xs text-gray-500">
            <summary className="cursor-pointer">Can't scan? Enter the key manually</summary>
            <code className="block mt-2 p-2 bg-gray-100 dark:bg-gray-800 rounded break-all">
              {totpEnrollment.secret}
            </code>
          </details>
          <div className="flex gap-2">
            <button className="btn-primary" onClick={() => setStep('confirm')}>I've added it →</button>
            <button className="btn-secondary" onClick={() => setStep('status')}>Cancel</button>
          </div>
        </section>
      )}

      {step === 'confirm' && (
        <section className="space-y-4">
          <h2 className="text-lg font-medium">Step 3 — Confirm</h2>

          <div>
            <label className="block text-sm font-medium mb-1">Daily loss cap (USD)</label>
            <input
              type="number"
              min={1}
              max={1000}
              value={dailyCap}
              onChange={(e) => setDailyCap(Number(e.target.value))}
              className="input"
            />
            <p className="text-xs text-gray-500 mt-1">
              Beta cap: $1,000. Live trading auto-reverts to paper when today's realised losses reach this value.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">
              Type this phrase exactly:&nbsp;
              <code>{REQUIRED_RISK_PHRASE}</code>
            </label>
            <input
              type="text"
              value={riskPhrase}
              onChange={(e) => setRiskPhrase(e.target.value)}
              className="input"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">6-digit code from your authenticator</label>
            <input
              type="text"
              inputMode="numeric"
              pattern="\d{6}"
              maxLength={6}
              value={totpCode}
              onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
              className="input font-mono tracking-widest"
              placeholder="123456"
            />
          </div>

          <div className="flex gap-2">
            <button className="btn-primary" disabled={!confirmReady || busy} onClick={handleEnable}>
              {busy ? 'Enabling…' : 'Enable live trading'}
            </button>
            <button className="btn-secondary" onClick={() => setStep('status')}>Cancel</button>
          </div>
        </section>
      )}

      {step === 'success' && (
        <section className="space-y-4">
          <h2 className="text-lg font-medium text-green-700 dark:text-green-400">
            Live trading enabled
          </h2>
          <p className="text-sm">
            Every order from now on will require your 6-digit code and route to a
            real Alpaca account. Losses are capped at ${dailyCap}/day.
          </p>
          <button className="btn-secondary" onClick={() => setStep('status')}>Back to status</button>
        </section>
      )}
    </div>
  );
}

function StatusPanel({
  status,
  onEnable,
  onDisable,
  busy,
}: {
  status: LiveModeStatus | null;
  onEnable: () => void;
  onDisable: () => void;
  busy: boolean;
}) {
  if (!status) {
    return <div>Loading…</div>;
  }
  const isLive = status.trading_mode === 'live';
  return (
    <section className="space-y-4">
      <div
        className={`rounded border p-4 ${
          isLive
            ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
            : 'border-gray-300 dark:border-gray-700'
        }`}
      >
        <div className="font-medium">
          Current mode:&nbsp;
          <span className={isLive ? 'text-green-700 dark:text-green-400' : ''}>
            {isLive ? 'LIVE' : 'Paper'}
          </span>
        </div>
        {isLive && status.daily_loss_cap_usd !== null && (
          <div className="text-sm text-gray-600 dark:text-gray-400">
            Daily loss cap: ${status.daily_loss_cap_usd}
          </div>
        )}
      </div>
      {isLive ? (
        <button className="btn-secondary" onClick={onDisable} disabled={busy}>
          Switch back to paper
        </button>
      ) : (
        <button className="btn-primary" onClick={onEnable} disabled={busy}>
          Enable live trading
        </button>
      )}
    </section>
  );
}

function LabeledInput({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <label className="block">
      <span className="block text-sm font-medium mb-1">{label}</span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="input"
      />
    </label>
  );
}

function Checkbox({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-2 text-sm">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1"
      />
      <span>{label}</span>
    </label>
  );
}

function extractError(e: unknown, fallback: string): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const resp = (e as { response?: { data?: { detail?: string } } }).response;
    if (resp?.data?.detail) return resp.data.detail;
  }
  if (e instanceof Error) return e.message;
  return fallback;
}
