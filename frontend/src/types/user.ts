export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string;
  risk_tolerance: string;
  investment_goal: string;
  notification_preferences: Record<string, boolean>;
  is_active: boolean;
  created_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  full_name: string;
}

export interface UpdateUserRequest {
  full_name?: string;
  email?: string;
  risk_tolerance?: string;
  investment_goal?: string;
  notification_preferences?: Record<string, boolean>;
}
