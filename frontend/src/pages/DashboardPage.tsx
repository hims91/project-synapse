import React from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, 
  Globe, 
  Database, 
  Zap, 
  Activity,
  Clock,
  Users,
  BarChart3
} from 'lucide-react'

import { useQuery } from '@tanstack/react-query'
import { analyticsApi } from '../lib/api'
import { formatNumber, formatRelativeTime } from '../lib/utils'
import LoadingSpinner from '../components/ui/LoadingSpinner'

const stats = [
  {
    name: 'Articles Processed',
    value: '12,847',
    change: '+12%',
    changeType: 'positive',
    icon: Database,
  },
  {
    name: 'Active Feeds',
    value: '234',
    change: '+3%',
    changeType: 'positive',
    icon: Globe,
  },
  {
    name: 'API Requests',
    value: '89.2K',
    change: '+18%',
    changeType: 'positive',
    icon: Zap,
  },
  {
    name: 'Processing Speed',
    value: '1.2s',
    change: '-8%',
    changeType: 'positive',
    icon: Activity,
  },
]

const recentActivity = [
  {
    id: 1,
    type: 'scrape',
    title: 'New article scraped from TechCrunch',
    time: '2 minutes ago',
    status: 'completed',
  },
  {
    id: 2,
    type: 'analysis',
    title: 'Bias analysis completed for 15 articles',
    time: '5 minutes ago',
    status: 'completed',
  },
  {
    id: 3,
    type: 'feed',
    title: 'RSS feed updated: Hacker News',
    time: '12 minutes ago',
    status: 'completed',
  },
  {
    id: 4,
    type: 'api',
    title: 'API rate limit warning for user',
    time: '18 minutes ago',
    status: 'warning',
  },
]

export default function DashboardPage() {
  const { data: dashboardStats, isLoading } = useQuery({
    queryKey: ['dashboard', 'stats'],
    queryFn: analyticsApi.getDashboardStats,
    refetchInterval: 30000, // Refetch every 30 seconds
  })

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-neural-900 dark:text-neural-100">
          Dashboard
        </h1>
        <p className="text-neural-600 dark:text-neural-400 mt-2">
          Welcome back! Here's what's happening with your data pipeline.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon
          return (
            <motion.div
              key={stat.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="card card-hover p-6"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-neural-600 dark:text-neural-400">
                    {stat.name}
                  </p>
                  <p className="text-2xl font-bold text-neural-900 dark:text-neural-100 mt-2">
                    {stat.value}
                  </p>
                  <div className="flex items-center mt-2">
                    <span
                      className={`text-sm font-medium ${
                        stat.changeType === 'positive'
                          ? 'text-success-600 dark:text-success-400'
                          : 'text-error-600 dark:text-error-400'
                      }`}
                    >
                      {stat.change}
                    </span>
                    <span className="text-sm text-neural-500 dark:text-neural-400 ml-1">
                      from last week
                    </span>
                  </div>
                </div>
                <div className="w-12 h-12 bg-synapse-100 dark:bg-synapse-900/30 rounded-lg flex items-center justify-center">
                  <Icon className="w-6 h-6 text-synapse-600 dark:text-synapse-400" />
                </div>
              </div>
            </motion.div>
          )
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
              Recent Activity
            </h2>
            <Clock className="w-5 h-5 text-neural-400" />
          </div>

          <div className="space-y-4">
            {recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-start space-x-3">
                <div
                  className={`w-2 h-2 rounded-full mt-2 ${
                    activity.status === 'completed'
                      ? 'bg-success-500'
                      : activity.status === 'warning'
                      ? 'bg-warning-500'
                      : 'bg-neural-300'
                  }`}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-neural-900 dark:text-neural-100">
                    {activity.title}
                  </p>
                  <p className="text-xs text-neural-500 dark:text-neural-400 mt-1">
                    {activity.time}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <button className="w-full mt-4 text-sm text-synapse-600 dark:text-synapse-400 hover:text-synapse-700 dark:hover:text-synapse-300 font-medium">
            View all activity
          </button>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="card p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
              Quick Actions
            </h2>
            <Zap className="w-5 h-5 text-neural-400" />
          </div>

          <div className="space-y-3">
            <button className="w-full btn btn-primary btn-md justify-start">
              <Globe className="w-4 h-4 mr-3" />
              Scrape New URL
            </button>
            <button className="w-full btn btn-secondary btn-md justify-start">
              <Database className="w-4 h-4 mr-3" />
              Explore Data
            </button>
            <button className="w-full btn btn-secondary btn-md justify-start">
              <BarChart3 className="w-4 h-4 mr-3" />
              View Analytics
            </button>
            <button className="w-full btn btn-secondary btn-md justify-start">
              <Users className="w-4 h-4 mr-3" />
              API Playground
            </button>
          </div>
        </motion.div>
      </div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.6 }}
        className="card p-6"
      >
        <h2 className="text-lg font-semibold text-neural-900 dark:text-neural-100 mb-6">
          System Status
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 bg-success-100 dark:bg-success-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
              <div className="w-3 h-3 bg-success-500 rounded-full animate-pulse" />
            </div>
            <h3 className="font-medium text-neural-900 dark:text-neural-100">
              API Services
            </h3>
            <p className="text-sm text-success-600 dark:text-success-400 mt-1">
              All systems operational
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 bg-success-100 dark:bg-success-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
              <div className="w-3 h-3 bg-success-500 rounded-full animate-pulse" />
            </div>
            <h3 className="font-medium text-neural-900 dark:text-neural-100">
              Data Pipeline
            </h3>
            <p className="text-sm text-success-600 dark:text-success-400 mt-1">
              Processing normally
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 bg-warning-100 dark:bg-warning-900/30 rounded-full flex items-center justify-center mx-auto mb-3">
              <div className="w-3 h-3 bg-warning-500 rounded-full animate-pulse" />
            </div>
            <h3 className="font-medium text-neural-900 dark:text-neural-100">
              Feed Polling
            </h3>
            <p className="text-sm text-warning-600 dark:text-warning-400 mt-1">
              Minor delays detected
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}