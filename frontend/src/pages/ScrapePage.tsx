import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Globe, 
  Zap, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Play,
  Pause,
  RotateCcw,
  ExternalLink,
  Eye,
  Download
} from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'

import { scrapeApi } from '../lib/api'
import { formatRelativeTime, isValidUrl, cn } from '../lib/utils'
import LoadingSpinner from '../components/ui/LoadingSpinner'

const scrapeSchema = z.object({
  url: z.string().url('Please enter a valid URL'),
  priority: z.boolean().default(false),
  enableBiasAnalysis: z.boolean().default(false),
  enableSummarization: z.boolean().default(true),
  enableEntityExtraction: z.boolean().default(true),
})

type ScrapeForm = z.infer<typeof scrapeSchema>

interface ScrapeJob {
  id: string
  url: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  priority: boolean
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  result?: {
    article_id?: string
    title?: string
    word_count?: number
    processing_time?: number
  }
}

// Mock jobs for demonstration
const mockJobs: ScrapeJob[] = [
  {
    id: '1',
    url: 'https://techcrunch.com/ai-breakthrough',
    status: 'completed',
    priority: true,
    created_at: '2024-01-15T10:30:00Z',
    started_at: '2024-01-15T10:30:05Z',
    completed_at: '2024-01-15T10:30:45Z',
    result: {
      article_id: 'art_123',
      title: 'Major AI Breakthrough Announced',
      word_count: 1250,
      processing_time: 40
    }
  },
  {
    id: '2',
    url: 'https://example.com/long-article',
    status: 'processing',
    priority: false,
    created_at: '2024-01-15T11:00:00Z',
    started_at: '2024-01-15T11:00:10Z',
  },
  {
    id: '3',
    url: 'https://invalid-site.com/article',
    status: 'failed',
    priority: false,
    created_at: '2024-01-15T09:45:00Z',
    started_at: '2024-01-15T09:45:05Z',
    completed_at: '2024-01-15T09:45:15Z',
    error_message: 'Failed to fetch content: Site returned 404'
  }
]

export default function ScrapePage() {
  const [jobs, setJobs] = useState<ScrapeJob[]>(mockJobs)
  const [selectedJob, setSelectedJob] = useState<ScrapeJob | null>(null)
  const queryClient = useQueryClient()

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<ScrapeForm>({
    resolver: zodResolver(scrapeSchema),
  })

  // Mock WebSocket connection for real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setJobs(prevJobs => 
        prevJobs.map(job => {
          if (job.status === 'processing' && Math.random() > 0.7) {
            return {
              ...job,
              status: 'completed',
              completed_at: new Date().toISOString(),
              result: {
                article_id: `art_${Date.now()}`,
                title: 'Scraped Article Title',
                word_count: Math.floor(Math.random() * 2000) + 500,
                processing_time: Math.floor(Math.random() * 60) + 10
              }
            }
          }
          return job
        })
      )
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const submitJobMutation = useMutation({
    mutationFn: async (data: ScrapeForm) => {
      // Mock API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const newJob: ScrapeJob = {
        id: Date.now().toString(),
        url: data.url,
        status: 'pending',
        priority: data.priority,
        created_at: new Date().toISOString(),
      }
      
      setJobs(prev => [newJob, ...prev])
      
      // Simulate job progression
      setTimeout(() => {
        setJobs(prev => prev.map(job => 
          job.id === newJob.id 
            ? { ...job, status: 'processing', started_at: new Date().toISOString() }
            : job
        ))
      }, 2000)
      
      return newJob
    },
    onSuccess: () => {
      toast.success('Scraping job submitted successfully!')
      reset()
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to submit scraping job')
    },
  })

  const onSubmit = (data: ScrapeForm) => {
    submitJobMutation.mutate(data)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4 text-warning-500" />
      case 'processing':
        return <LoadingSpinner size="sm" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-success-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-error-500" />
      default:
        return <AlertCircle className="w-4 h-4 text-neural-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'badge-warning'
      case 'processing':
        return 'badge-primary'
      case 'completed':
        return 'badge-success'
      case 'failed':
        return 'badge-error'
      default:
        return 'badge-neutral'
    }
  }

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
          Scrape Content
        </h1>
        <p className="text-neural-600 dark:text-neural-400 mt-2">
          Submit URLs for intelligent content extraction and analysis.
        </p>
      </div>

      {/* URL Input Form */}
      <div className="card p-6">
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-neural-700 dark:text-neural-300 mb-2">
              URL to Scrape
            </label>
            <div className="flex space-x-3">
              <div className="flex-1">
                <input
                  {...register('url')}
                  type="url"
                  placeholder="https://example.com/article"
                  className={cn('input', errors.url && 'border-error-500')}
                />
                {errors.url && (
                  <p className="mt-1 text-sm text-error-600 dark:text-error-400">
                    {errors.url.message}
                  </p>
                )}
              </div>
              <button 
                type="submit" 
                disabled={isSubmitting}
                className="btn btn-primary btn-md"
              >
                {isSubmitting ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : (
                  <Globe className="w-4 h-4 mr-2" />
                )}
                {isSubmitting ? 'Submitting...' : 'Scrape'}
              </button>
            </div>
          </div>
          
          {/* Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="flex items-center">
              <input 
                {...register('priority')}
                type="checkbox" 
                className="rounded border-neural-300 text-synapse-600 focus:ring-synapse-500" 
              />
              <span className="ml-2 text-sm text-neural-700 dark:text-neural-300">
                High Priority Queue
              </span>
            </label>
            <label className="flex items-center">
              <input 
                {...register('enableBiasAnalysis')}
                type="checkbox" 
                className="rounded border-neural-300 text-synapse-600 focus:ring-synapse-500" 
              />
              <span className="ml-2 text-sm text-neural-700 dark:text-neural-300">
                Enable Bias Analysis
              </span>
            </label>
            <label className="flex items-center">
              <input 
                {...register('enableSummarization')}
                type="checkbox" 
                className="rounded border-neural-300 text-synapse-600 focus:ring-synapse-500" 
              />
              <span className="ml-2 text-sm text-neural-700 dark:text-neural-300">
                Generate Summary
              </span>
            </label>
            <label className="flex items-center">
              <input 
                {...register('enableEntityExtraction')}
                type="checkbox" 
                className="rounded border-neural-300 text-synapse-600 focus:ring-synapse-500" 
              />
              <span className="ml-2 text-sm text-neural-700 dark:text-neural-300">
                Extract Entities
              </span>
            </label>
          </div>
        </form>
      </div>

      {/* Job Queue Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: 'Pending', count: jobs.filter(j => j.status === 'pending').length, color: 'text-warning-600' },
          { label: 'Processing', count: jobs.filter(j => j.status === 'processing').length, color: 'text-synapse-600' },
          { label: 'Completed', count: jobs.filter(j => j.status === 'completed').length, color: 'text-success-600' },
          { label: 'Failed', count: jobs.filter(j => j.status === 'failed').length, color: 'text-error-600' },
        ].map((stat) => (
          <div key={stat.label} className="card p-4 text-center">
            <div className={`text-2xl font-bold ${stat.color}`}>
              {stat.count}
            </div>
            <div className="text-sm text-neural-600 dark:text-neural-400">
              {stat.label}
            </div>
          </div>
        ))}
      </div>

      {/* Job History */}
      <div className="card">
        <div className="p-6 border-b border-neural-200 dark:border-neural-700">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
              Recent Jobs
            </h2>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-neural-500 dark:text-neural-400">
                Live updates
              </span>
            </div>
          </div>
        </div>

        <div className="divide-y divide-neural-200 dark:divide-neural-700">
          {jobs.map((job) => (
            <motion.div
              key={job.id}
              layout
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-6 hover:bg-neutral-50 dark:hover:bg-neural-800/50 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(job.status)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <h3 className="font-medium text-neural-900 dark:text-neural-100 truncate">
                          {job.result?.title || new URL(job.url).hostname}
                        </h3>
                        {job.priority && (
                          <span className="badge badge-warning text-xs">Priority</span>
                        )}
                        <span className={`badge text-xs ${getStatusColor(job.status)}`}>
                          {job.status}
                        </span>
                      </div>
                      <p className="text-sm text-neural-500 dark:text-neural-400 truncate">
                        {job.url}
                      </p>
                      <div className="flex items-center space-x-4 mt-1 text-xs text-neural-400">
                        <span>Created {formatRelativeTime(job.created_at)}</span>
                        {job.completed_at && (
                          <span>Completed {formatRelativeTime(job.completed_at)}</span>
                        )}
                        {job.result?.processing_time && (
                          <span>{job.result.processing_time}s processing</span>
                        )}
                        {job.result?.word_count && (
                          <span>{job.result.word_count} words</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Error Message */}
                  {job.error_message && (
                    <div className="mt-2 p-2 bg-error-50 dark:bg-error-900/20 rounded text-sm text-error-700 dark:text-error-300">
                      {job.error_message}
                    </div>
                  )}

                  {/* Progress Bar for Processing Jobs */}
                  {job.status === 'processing' && (
                    <div className="mt-2">
                      <div className="w-full bg-neural-200 dark:bg-neural-700 rounded-full h-1">
                        <div className="bg-synapse-600 h-1 rounded-full animate-pulse" style={{ width: '60%' }}></div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-2 ml-4">
                  {job.status === 'completed' && job.result?.article_id && (
                    <>
                      <button className="btn btn-ghost btn-sm">
                        <Eye className="w-3 h-3" />
                      </button>
                      <button className="btn btn-ghost btn-sm">
                        <Download className="w-3 h-3" />
                      </button>
                    </>
                  )}
                  <button className="btn btn-ghost btn-sm">
                    <ExternalLink className="w-3 h-3" />
                  </button>
                  {job.status === 'failed' && (
                    <button className="btn btn-ghost btn-sm">
                      <RotateCcw className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {jobs.length === 0 && (
          <div className="p-12 text-center">
            <Globe className="w-12 h-12 text-neural-300 dark:text-neural-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-neural-900 dark:text-neural-100 mb-2">
              No scraping jobs yet
            </h3>
            <p className="text-neural-500 dark:text-neural-400">
              Submit your first URL above to get started with content extraction.
            </p>
          </div>
        )}
      </div>
    </motion.div>
  )
}