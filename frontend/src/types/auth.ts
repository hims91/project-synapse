export interface User {
  id: string
  name: string
  email: string
  tier: 'free' | 'premium' | 'enterprise'
  avatar?: string
  createdAt: string
  lastLoginAt?: string
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface AuthResponse {
  user: User
  token: string
  expiresAt: string
}