import { apiClient } from './client'

export interface LoginCredentials {
  username: string
  password: string
}

export interface RegisterData {
  username: string
  email: string
  password: string
  full_name?: string
}

export interface AuthResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export const authApi = {
  login: async (credentials: LoginCredentials): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/auth/login', credentials)
    return response.data
  },

  register: async (data: RegisterData) => {
    const response = await apiClient.post('/auth/register', data)
    return response.data
  },

  refreshToken: async (refreshToken: string): Promise<AuthResponse> => {
    const response = await apiClient.post<AuthResponse>('/auth/refresh', {
      refresh_token: refreshToken,
    })
    return response.data
  },
}

