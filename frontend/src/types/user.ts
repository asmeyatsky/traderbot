export interface User {
  id: string;
  email: string;
  first_name: string;
  last_name: string;
  risk_tolerance: string;
  investment_goal: string;
  max_position_size_percentage: number;
  daily_loss_limit: number | null;
  weekly_loss_limit: number | null;
  monthly_loss_limit: number | null;
  sector_preferences: string[];
  sector_exclusions: string[];
  is_active: boolean;
  email_notifications_enabled: boolean;
  sms_notifications_enabled: boolean;
  approval_mode_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RegisterRequest {
  email: string;
  first_name: string;
  last_name: string;
  password: string;
  risk_tolerance: string;
  investment_goal: string;
}

export interface UpdateUserRequest {
  first_name?: string;
  last_name?: string;
  risk_tolerance?: string;
  investment_goal?: string;
  email_notifications_enabled?: boolean;
  sms_notifications_enabled?: boolean;
  approval_mode_enabled?: boolean;
}
