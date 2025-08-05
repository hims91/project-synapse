import React from 'react'
import { motion } from 'framer-motion'
import { Settings, User, Key, Bell, Shield } from 'lucide-react'

export default function SettingsPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-8"
    >
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-neural-900 dark:text-neural-100">
          Settings
        </h1>
        <p className="text-neural-600 dark:text-neural-400 mt-2">
          Manage your account, preferences, and system configuration.
        </p>
      </div>

      {/* Settings Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          {
            title: 'Profile Settings',
            description: 'Update your personal information and preferences',
            icon: User,
            color: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
          },
          {
            title: 'API Keys',
            description: 'Manage your API keys and access tokens',
            icon: Key,
            color: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
          },
          {
            title: 'Notifications',
            description: 'Configure alerts and notification preferences',
            icon: Bell,
            color: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400',
          },
          {
            title: 'Security',
            description: 'Security settings and two-factor authentication',
            icon: Shield,
            color: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
          },
        ].map((item) => {
          const Icon = item.icon
          return (
            <div key={item.title} className="card card-hover p-6">
              <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${item.color}`}>
                <Icon className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-neural-900 dark:text-neural-100 mb-2">
                {item.title}
              </h3>
              <p className="text-sm text-neural-600 dark:text-neural-400 mb-4">
                {item.description}
              </p>
              <button className="btn btn-secondary btn-sm">
                Configure
              </button>
            </div>
          )
        })}
      </div>

      {/* Coming Soon */}
      <div className="card p-12 text-center">
        <div className="w-16 h-16 bg-synapse-100 dark:bg-synapse-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
          <Settings className="w-8 h-8 text-synapse-600 dark:text-synapse-400" />
        </div>
        <h2 className="text-xl font-semibold text-neural-900 dark:text-neural-100 mb-2">
          Advanced Settings Coming Soon
        </h2>
        <p className="text-neural-600 dark:text-neural-400 max-w-md mx-auto">
          Comprehensive settings management, user preferences, and system configuration
          options are being developed.
        </p>
      </div>
    </motion.div>
  )
}