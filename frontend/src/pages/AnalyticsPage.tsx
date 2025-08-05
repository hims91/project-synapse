import React from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, PieChart, Activity } from 'lucide-react'

export default function AnalyticsPage() {
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
          Analytics
        </h1>
        <p className="text-neural-600 dark:text-neural-400 mt-2">
          Deep insights into your content pipeline and system performance.
        </p>
      </div>

      {/* Analytics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          {
            title: 'Content Trends',
            description: 'Track trending topics and content velocity',
            icon: TrendingUp,
            color: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
          },
          {
            title: 'Source Analysis',
            description: 'Analyze content sources and feed performance',
            icon: PieChart,
            color: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
          },
          {
            title: 'Processing Metrics',
            description: 'Monitor system performance and processing times',
            icon: Activity,
            color: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
          },
          {
            title: 'API Usage',
            description: 'Track API endpoint usage and performance',
            icon: BarChart3,
            color: 'bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400',
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
              <p className="text-sm text-neural-600 dark:text-neural-400">
                {item.description}
              </p>
            </div>
          )
        })}
      </div>

      {/* Coming Soon */}
      <div className="card p-12 text-center">
        <div className="w-16 h-16 bg-synapse-100 dark:bg-synapse-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
          <BarChart3 className="w-8 h-8 text-synapse-600 dark:text-synapse-400" />
        </div>
        <h2 className="text-xl font-semibold text-neural-900 dark:text-neural-100 mb-2">
          Advanced Analytics Coming Soon
        </h2>
        <p className="text-neural-600 dark:text-neural-400 max-w-md mx-auto">
          Interactive charts, real-time metrics, and comprehensive reporting dashboards
          are being developed to provide deep insights into your data pipeline.
        </p>
      </div>
    </motion.div>
  )
}