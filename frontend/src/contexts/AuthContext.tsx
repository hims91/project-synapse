import React, { createContext, useContext, useEffect, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'

import { authApi } from '../lib/api'
import type { User, LoginCredentials } from '../types/auth'

interface AuthContextType {
  user: User | null
  isLoading: boolean
  login: (credentials: LoginCredentials) => Promise<void>
  logout: () => void
  refreshUser: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const queryClient = useQueryClient()

  // Check for existing session on mount
  const { data: userData, isLoading } = useQuery({
    queryKey: ['auth', 'user'],
    queryFn: authApi.getCurrentUser,
    retry: false,
    staleTime: 1000 * 60 * 5, // 5 minutes
  })

  useEffect(() => {
    if (userData) {
      setUser(userData)
    }
  }, [userData])

  const loginMutation = useMutation({
    mutationFn: authApi.login,
    onSuccess: (data) => {
      setUser(data.user)
      localStorage.setItem('synapse-token', data.token)
      queryClient.setQueryData(['auth', 'user'], data.user)
      toast.success('Welcome to Project Synapse!')
    },
    onError: (error: any) => {
      const message = error?.response?.data?.detail || 'Login failed'
      toast.error(message)
    },
  })

  const login = async (credentials: LoginCredentials) => {
    await loginMutation.mutateAsync(credentials)
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem('synapse-token')
    queryClient.clear()
    toast.success('Logged out successfully')
  }

  const refreshUser = () => {
    queryClient.invalidateQueries({ queryKey: ['auth', 'user'] })
  }

  const value = {
    user,
    isLoading,
    login,
    logout,
    refreshUser,
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}