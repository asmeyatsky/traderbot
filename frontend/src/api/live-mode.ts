import apiClient from './client';

export interface TotpEnrollmentResponse {
  secret: string;
  provisioning_uri: string;
  issuer: string;
}

export interface LiveModeStatus {
  trading_mode: 'paper' | 'live';
  daily_loss_cap_usd: number | null;
  live_mode_enabled_at: string | null;
  has_totp: boolean;
}

export interface EnableLiveModeRequest {
  kyc_attestation_payload: string;
  daily_loss_cap_usd: number;
  totp_code: string;
  risk_acknowledgement: string;
}

const PREFIX = '/api/v1/users/me';

export async function enrollTotp(): Promise<TotpEnrollmentResponse> {
  const { data } = await apiClient.post<TotpEnrollmentResponse>(
    `${PREFIX}/totp/enroll`,
  );
  return data;
}

export async function enableLiveMode(
  payload: EnableLiveModeRequest,
): Promise<LiveModeStatus> {
  const { data } = await apiClient.post<LiveModeStatus>(
    `${PREFIX}/enable-live-mode`,
    payload,
  );
  return data;
}

export async function disableLiveMode(): Promise<LiveModeStatus> {
  const { data } = await apiClient.post<LiveModeStatus>(
    `${PREFIX}/disable-live-mode`,
  );
  return data;
}

export async function getLiveModeStatus(): Promise<LiveModeStatus> {
  const { data } = await apiClient.get<LiveModeStatus>(
    `${PREFIX}/live-mode-status`,
  );
  return data;
}

// Exact phrase required by backend — kept in sync with REQUIRED_RISK_PHRASE
// in src/presentation/api/routers/users.py.
export const REQUIRED_RISK_PHRASE = 'I understand I will lose real money.';
