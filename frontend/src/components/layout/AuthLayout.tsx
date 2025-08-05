import React from 'react'
import { motion } from 'framer-motion'

interface AuthLayoutProps {
  children: React.ReactNode
}

export default function AuthLayout({ children }: AuthLayoutProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-synapse-50 to-synapse-100 dark:from-neural-950 dark:to-neural-900">
      <div className="max-w-md w-full mx-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white dark:bg-neural-800 rounded-2xl shadow-xl p-8"
        >
          {/* Logo */}
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-synapse-500 to-synapse-600 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <span className="text-white font-bold text-2xl">S</span>
            </div>
            <h1 className="text-2xl font-bold text-neural-900 dark:text-neural-100">
              Project Synapse
            </h1>
            <p className="text-neural-600 dark:text-neural-400 text-sm mt-1">
              Feel the web. Think in data. Act with insight.
            </p>
          </div>

          {children}
        </motion.div>
      </div>
    </div>
  )
}