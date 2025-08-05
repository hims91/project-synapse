import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Eye, EyeOff, LogIn } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

import { useAuth } from '../../contexts/AuthContext'
import LoadingSpinner from '../../components/ui/LoadingSpinner'

const loginSchema = z.object({
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(1, 'Password is required'),
})

type LoginForm = z.infer<typeof loginSchema>

export default function LoginPage() {
  const [showPassword, setShowPassword] = useState(false)
  const { login } = useAuth()

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<LoginForm>({
    resolver: zodResolver(loginSchema),
  })

  const onSubmit = async (data: LoginForm) => {
    try {
      await login(data)
    } catch (error) {
      // Error is handled by the auth context
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-neural-700 dark:text-neural-300 mb-2">
            Email address
          </label>
          <input
            {...register('email')}
            type="email"
            id="email"
            className="input"
            placeholder="Enter your email"
          />
          {errors.email && (
            <p className="mt-1 text-sm text-error-600 dark:text-error-400">
              {errors.email.message}
            </p>
          )}
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-medium text-neural-700 dark:text-neural-300 mb-2">
            Password
          </label>
          <div className="relative">
            <input
              {...register('password')}
              type={showPassword ? 'text' : 'password'}
              id="password"
              className="input pr-10"
              placeholder="Enter your password"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-neural-400 hover:text-neural-600 dark:hover:text-neural-300"
            >
              {showPassword ? (
                <EyeOff className="w-4 h-4" />
              ) : (
                <Eye className="w-4 h-4" />
              )}
            </button>
          </div>
          {errors.password && (
            <p className="mt-1 text-sm text-error-600 dark:text-error-400">
              {errors.password.message}
            </p>
          )}
        </div>

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          type="submit"
          disabled={isSubmitting}
          className="w-full btn btn-primary btn-lg"
        >
          {isSubmitting ? (
            <LoadingSpinner size="sm" className="mr-2" />
          ) : (
            <LogIn className="w-4 h-4 mr-2" />
          )}
          {isSubmitting ? 'Signing in...' : 'Sign in'}
        </motion.button>
      </form>

      <div className="mt-8 p-4 bg-synapse-50 dark:bg-synapse-900/20 rounded-lg">
        <h3 className="text-sm font-medium text-synapse-800 dark:text-synapse-300 mb-2">
          Demo Credentials
        </h3>
        <div className="text-xs text-synapse-600 dark:text-synapse-400 space-y-1">
          <p><strong>Email:</strong> demo@synapse.dev</p>
          <p><strong>Password:</strong> demo</p>
        </div>
      </div>
    </div>
  )
}