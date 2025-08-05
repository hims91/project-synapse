import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Code, 
  Play, 
  Book, 
  Zap, 
  Send,
  Copy,
  Download,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  XCircle,
  Clock,
  Settings
} from 'lucide-react'
import { useForm } from 'react-hook-form'
import toast from 'react-hot-toast'

import { copyToClipboard, cn } from '../lib/utils'
import LoadingSpinner from '../components/ui/LoadingSpinner'

interface APIEndpoint {
  id: string
  method: 'GET' | 'POST' | 'PUT' | 'DELETE'
  path: string
  name: string
  description: string
  parameters?: Array<{
    name: string
    type: string
    required: boolean
    description: string
    example?: any
  }>
  requestBody?: {
    type: string
    properties: Record<string, any>
    example: any
  }
  responses: Record<string, any>
}

const apiCategories = [
  {
    name: 'FinMind API',
    description: 'Financial market intelligence and sentiment analysis',
    color: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
    endpoints: [
      {
        id: 'finmind-market',
        method: 'GET' as const,
        path: '/financial/market',
        name: 'Market Analysis',
        description: 'Analyze market sentiment with ticker filtering',
        parameters: [
          { name: 'tickers', type: 'string', required: false, description: 'Comma-separated ticker symbols', example: 'AAPL,GOOGL' },
          { name: 'sentiment_threshold', type: 'number', required: false, description: 'Minimum sentiment score', example: 0.0 },
          { name: 'days_back', type: 'number', required: false, description: 'Number of days to analyze', example: 7 },
        ],
        responses: {
          200: { description: 'Market analysis results' }
        }
      },
      {
        id: 'finmind-trends',
        method: 'GET' as const,
        path: '/financial/trends',
        name: 'Financial Trends',
        description: 'Analyze financial trends and correlations',
        parameters: [
          { name: 'sector', type: 'string', required: false, description: 'Market sector filter', example: 'tech' },
          { name: 'time_window', type: 'string', required: false, description: 'Time window for analysis', example: '7d' },
        ],
        responses: {
          200: { description: 'Financial trends analysis' }
        }
      }
    ]
  },
  {
    name: 'Digestify API',
    description: 'Advanced content summarization and quality scoring',
    color: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
    endpoints: [
      {
        id: 'digestify-summarize',
        method: 'POST' as const,
        path: '/summarize',
        name: 'Summarize Content',
        description: 'Generate extractive and abstractive summaries',
        requestBody: {
          type: 'object',
          properties: {
            text: { type: 'string', description: 'Text to summarize' },
            mode: { type: 'string', enum: ['extractive', 'abstractive', 'hybrid'] },
            length: { type: 'string', enum: ['short', 'medium', 'long'] }
          },
          example: {
            text: 'Your article content here...',
            mode: 'hybrid',
            length: 'medium'
          }
        },
        responses: {
          200: { description: 'Summarization results' }
        }
      }
    ]
  },
  {
    name: 'Narrative API',
    description: 'Bias detection and narrative analysis',
    color: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
    endpoints: [
      {
        id: 'narrative-analysis',
        method: 'POST' as const,
        path: '/analysis/narrative',
        name: 'Narrative Analysis',
        description: 'Comprehensive bias analysis and narrative extraction',
        requestBody: {
          type: 'object',
          properties: {
            text: { type: 'string', description: 'Text to analyze' },
            include_bias_indicators: { type: 'boolean', default: true },
            include_framing_analysis: { type: 'boolean', default: true }
          },
          example: {
            text: 'Your article content here...',
            include_bias_indicators: true,
            include_framing_analysis: true
          }
        },
        responses: {
          200: { description: 'Narrative analysis results' }
        }
      }
    ]
  }
]

export default function APIPlaygroundPage() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedEndpoint, setSelectedEndpoint] = useState<APIEndpoint | null>(null)
  const [requestData, setRequestData] = useState<any>({})
  const [response, setResponse] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())

  const { register, handleSubmit, reset } = useForm()

  const toggleCategory = (categoryName: string) => {
    const newExpanded = new Set(expandedCategories)
    if (newExpanded.has(categoryName)) {
      newExpanded.delete(categoryName)
    } else {
      newExpanded.add(categoryName)
    }
    setExpandedCategories(newExpanded)
  }

  const selectEndpoint = (endpoint: APIEndpoint) => {
    setSelectedEndpoint(endpoint)
    setResponse(null)
    
    // Initialize request data with example values
    const initialData: any = {}
    
    if (endpoint.parameters) {
      endpoint.parameters.forEach(param => {
        if (param.example !== undefined) {
          initialData[param.name] = param.example
        }
      })
    }
    
    if (endpoint.requestBody?.example) {
      Object.assign(initialData, endpoint.requestBody.example)
    }
    
    setRequestData(initialData)
  }

  const executeRequest = async (data: any) => {
    if (!selectedEndpoint) return

    setIsLoading(true)
    
    try {
      // Mock API call with realistic delay
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Generate mock response based on endpoint
      const mockResponse = generateMockResponse(selectedEndpoint)
      
      setResponse({
        status: 200,
        statusText: 'OK',
        data: mockResponse,
        headers: {
          'content-type': 'application/json',
          'x-response-time': '150ms'
        },
        executedAt: new Date().toISOString()
      })
      
      toast.success('Request executed successfully!')
    } catch (error) {
      setResponse({
        status: 500,
        statusText: 'Internal Server Error',
        data: { error: 'Mock API error for demonstration' },
        executedAt: new Date().toISOString()
      })
      toast.error('Request failed')
    } finally {
      setIsLoading(false)
    }
  }

  const generateMockResponse = (endpoint: APIEndpoint) => {
    switch (endpoint.id) {
      case 'finmind-market':
        return {
          analysis_period: {
            start_date: '2024-01-08T00:00:00Z',
            end_date: '2024-01-15T00:00:00Z',
            days_analyzed: 7
          },
          market_sentiment: 0.35,
          ticker_analysis: {
            'AAPL': { average_sentiment: 0.42, article_count: 15 },
            'GOOGL': { average_sentiment: 0.28, article_count: 12 }
          },
          market_pulse_score: 67.5
        }
      
      case 'digestify-summarize':
        return {
          summary: 'This is a generated summary of the provided content using advanced NLP techniques.',
          mode_used: 'hybrid',
          sentence_count: 3,
          word_count: 45,
          compression_ratio: 0.15,
          quality_score: 0.87,
          processing_time: 1.2
        }
      
      case 'narrative-analysis':
        return {
          overall_bias_score: 0.23,
          bias_indicators: [
            {
              type: 'confirmation',
              description: 'Detected confirmation bias through absolute language',
              confidence: 0.78,
              evidence: ['obviously', 'clearly']
            }
          ],
          framing_patterns: [
            {
              type: 'conflict_oriented',
              description: 'Detected conflict-oriented framing',
              confidence: 0.65,
              keywords: ['battle', 'versus']
            }
          ],
          confidence: 0.82
        }
      
      default:
        return { message: 'Mock response data', timestamp: new Date().toISOString() }
    }
  }

  const generateCurlCommand = () => {
    if (!selectedEndpoint) return ''

    const baseUrl = 'http://localhost:8000/api/v1'
    const url = `${baseUrl}${selectedEndpoint.path}`
    
    let curl = `curl -X ${selectedEndpoint.method} "${url}"`
    curl += ` \\\n  -H "Authorization: Bearer YOUR_API_KEY"`
    curl += ` \\\n  -H "Content-Type: application/json"`
    
    if (selectedEndpoint.method !== 'GET' && Object.keys(requestData).length > 0) {
      curl += ` \\\n  -d '${JSON.stringify(requestData, null, 2)}'`
    }
    
    return curl
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
          API Playground
        </h1>
        <p className="text-neural-600 dark:text-neural-400 mt-2">
          Test and explore all Project Synapse APIs with interactive documentation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* API Explorer Sidebar */}
        <div className="lg:col-span-1">
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-neural-900 dark:text-neural-100 mb-4">
              API Explorer
            </h2>
            
            <div className="space-y-2">
              {apiCategories.map((category) => (
                <div key={category.name}>
                  <button
                    onClick={() => toggleCategory(category.name)}
                    className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-neutral-100 dark:hover:bg-neural-700 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${category.color.split(' ')[0]}`} />
                      <span className="font-medium text-neural-900 dark:text-neural-100">
                        {category.name}
                      </span>
                    </div>
                    {expandedCategories.has(category.name) ? (
                      <ChevronDown className="w-4 h-4" />
                    ) : (
                      <ChevronRight className="w-4 h-4" />
                    )}
                  </button>
                  
                  <AnimatePresence>
                    {expandedCategories.has(category.name) && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="ml-6 mt-2 space-y-1"
                      >
                        {category.endpoints.map((endpoint) => (
                          <button
                            key={endpoint.id}
                            onClick={() => selectEndpoint(endpoint)}
                            className={cn(
                              'w-full text-left p-2 rounded text-sm transition-colors',
                              selectedEndpoint?.id === endpoint.id
                                ? 'bg-synapse-100 dark:bg-synapse-900/30 text-synapse-700 dark:text-synapse-300'
                                : 'hover:bg-neutral-100 dark:hover:bg-neural-700 text-neural-600 dark:text-neural-400'
                            )}
                          >
                            <div className="flex items-center space-x-2">
                              <span className={cn(
                                'px-1.5 py-0.5 rounded text-xs font-mono',
                                endpoint.method === 'GET' ? 'bg-green-100 text-green-700' :
                                endpoint.method === 'POST' ? 'bg-blue-100 text-blue-700' :
                                'bg-gray-100 text-gray-700'
                              )}>
                                {endpoint.method}
                              </span>
                              <span>{endpoint.name}</span>
                            </div>
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {selectedEndpoint ? (
            <>
              {/* Endpoint Details */}
              <div className="card p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <span className={cn(
                    'px-2 py-1 rounded text-sm font-mono',
                    selectedEndpoint.method === 'GET' ? 'bg-green-100 text-green-700' :
                    selectedEndpoint.method === 'POST' ? 'bg-blue-100 text-blue-700' :
                    'bg-gray-100 text-gray-700'
                  )}>
                    {selectedEndpoint.method}
                  </span>
                  <code className="text-sm bg-neural-100 dark:bg-neural-800 px-2 py-1 rounded">
                    {selectedEndpoint.path}
                  </code>
                </div>
                
                <h3 className="text-xl font-semibold text-neural-900 dark:text-neural-100 mb-2">
                  {selectedEndpoint.name}
                </h3>
                <p className="text-neural-600 dark:text-neural-400">
                  {selectedEndpoint.description}
                </p>
              </div>

              {/* Request Configuration */}
              <div className="card p-6">
                <h4 className="text-lg font-semibold text-neural-900 dark:text-neural-100 mb-4">
                  Request Configuration
                </h4>
                
                <form onSubmit={handleSubmit(executeRequest)} className="space-y-4">
                  {/* Parameters */}
                  {selectedEndpoint.parameters && selectedEndpoint.parameters.length > 0 && (
                    <div>
                      <h5 className="font-medium text-neural-900 dark:text-neural-100 mb-3">
                        Parameters
                      </h5>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {selectedEndpoint.parameters.map((param) => (
                          <div key={param.name}>
                            <label className="block text-sm font-medium text-neural-700 dark:text-neural-300 mb-1">
                              {param.name}
                              {param.required && <span className="text-error-500 ml-1">*</span>}
                            </label>
                            <input
                              type={param.type === 'number' ? 'number' : 'text'}
                              className="input"
                              placeholder={param.example?.toString() || param.description}
                              defaultValue={requestData[param.name] || ''}
                              onChange={(e) => setRequestData(prev => ({
                                ...prev,
                                [param.name]: param.type === 'number' ? Number(e.target.value) : e.target.value
                              }))}
                            />
                            <p className="text-xs text-neural-500 dark:text-neural-400 mt-1">
                              {param.description}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Request Body */}
                  {selectedEndpoint.requestBody && (
                    <div>
                      <h5 className="font-medium text-neural-900 dark:text-neural-100 mb-3">
                        Request Body
                      </h5>
                      <textarea
                        className="input min-h-[120px] font-mono text-sm"
                        placeholder="Enter JSON request body..."
                        value={JSON.stringify(requestData, null, 2)}
                        onChange={(e) => {
                          try {
                            setRequestData(JSON.parse(e.target.value))
                          } catch {
                            // Invalid JSON, keep as string for now
                          }
                        }}
                      />
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={isLoading}
                    className="btn btn-primary btn-md"
                  >
                    {isLoading ? (
                      <LoadingSpinner size="sm" className="mr-2" />
                    ) : (
                      <Send className="w-4 h-4 mr-2" />
                    )}
                    {isLoading ? 'Executing...' : 'Execute Request'}
                  </button>
                </form>
              </div>

              {/* Code Generation */}
              <div className="card p-6">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
                    Generated Code
                  </h4>
                  <button
                    onClick={() => copyToClipboard(generateCurlCommand())}
                    className="btn btn-ghost btn-sm"
                  >
                    <Copy className="w-3 h-3 mr-1" />
                    Copy cURL
                  </button>
                </div>
                
                <div className="bg-neural-900 dark:bg-neural-950 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-sm text-neural-100">
                    <code>{generateCurlCommand()}</code>
                  </pre>
                </div>
              </div>

              {/* Response */}
              {response && (
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
                      Response
                    </h4>
                    <div className="flex items-center space-x-2">
                      {response.status === 200 ? (
                        <CheckCircle className="w-4 h-4 text-success-500" />
                      ) : (
                        <XCircle className="w-4 h-4 text-error-500" />
                      )}
                      <span className={cn(
                        'text-sm font-medium',
                        response.status === 200 ? 'text-success-600' : 'text-error-600'
                      )}>
                        {response.status} {response.statusText}
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-neural-900 dark:bg-neural-950 rounded-lg p-4 overflow-x-auto">
                    <pre className="text-sm text-neural-100">
                      <code>{JSON.stringify(response.data, null, 2)}</code>
                    </pre>
                  </div>
                  
                  {response.headers && (
                    <div className="mt-4 text-xs text-neural-500 dark:text-neural-400">
                      Response time: {response.headers['x-response-time']} | 
                      Executed at: {new Date(response.executedAt).toLocaleTimeString()}
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            /* No Endpoint Selected */
            <div className="card p-12 text-center">
              <div className="w-16 h-16 bg-synapse-100 dark:bg-synapse-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <Code className="w-8 h-8 text-synapse-600 dark:text-synapse-400" />
              </div>
              <h3 className="text-xl font-semibold text-neural-900 dark:text-neural-100 mb-2">
                Select an API Endpoint
              </h3>
              <p className="text-neural-600 dark:text-neural-400">
                Choose an endpoint from the sidebar to start testing and exploring the API.
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}