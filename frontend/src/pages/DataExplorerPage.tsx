import React, { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Database, 
  Search, 
  Filter, 
  Download, 
  Code, 
  Copy, 
  Eye, 
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Calendar,
  Tag,
  Globe
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import toast from 'react-hot-toast'

import { contentApi } from '../lib/api'
import { formatDate, formatRelativeTime, copyToClipboard, cn } from '../lib/utils'
import LoadingSpinner from '../components/ui/LoadingSpinner'

// Mock data for demonstration
const mockArticles = [
  {
    id: '1',
    title: 'The Future of AI in Healthcare: Revolutionary Changes Ahead',
    url: 'https://example.com/ai-healthcare',
    content: 'Artificial intelligence is transforming healthcare at an unprecedented pace...',
    summary: 'AI technologies are revolutionizing healthcare delivery and patient outcomes.',
    author: 'Dr. Sarah Johnson',
    published_at: '2024-01-15T10:30:00Z',
    scraped_at: '2024-01-15T11:00:00Z',
    source_domain: 'healthtech.com',
    nlp_data: {
      sentiment: 0.7,
      entities: [
        { text: 'AI', label: 'TECHNOLOGY', confidence: 0.95 },
        { text: 'healthcare', label: 'INDUSTRY', confidence: 0.92 }
      ],
      categories: ['technology', 'healthcare', 'ai'],
      significance: 0.85
    },
    page_metadata: {
      word_count: 1250,
      reading_time: 5,
      og_image: 'https://example.com/ai-image.jpg'
    }
  },
  {
    id: '2',
    title: 'Climate Change Impact on Global Food Security',
    url: 'https://example.com/climate-food',
    content: 'Recent studies show alarming trends in global food production...',
    summary: 'Climate change poses significant threats to worldwide food security.',
    author: 'Environmental Research Team',
    published_at: '2024-01-14T14:20:00Z',
    scraped_at: '2024-01-14T15:00:00Z',
    source_domain: 'climatewatch.org',
    nlp_data: {
      sentiment: -0.3,
      entities: [
        { text: 'climate change', label: 'PHENOMENON', confidence: 0.98 },
        { text: 'food security', label: 'CONCEPT', confidence: 0.94 }
      ],
      categories: ['environment', 'climate', 'food'],
      significance: 0.92
    },
    page_metadata: {
      word_count: 2100,
      reading_time: 8,
      og_image: 'https://example.com/climate-image.jpg'
    }
  }
]

interface CodeSnippet {
  language: string
  code: string
  label: string
}

export default function DataExplorerPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedFilters, setSelectedFilters] = useState<string[]>([])
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())
  const [showCodeModal, setShowCodeModal] = useState(false)
  const [selectedArticle, setSelectedArticle] = useState<any>(null)

  // Mock query for demonstration
  const { data: articles = mockArticles, isLoading } = useQuery({
    queryKey: ['articles', searchQuery, selectedFilters],
    queryFn: () => Promise.resolve(mockArticles),
    enabled: true,
  })

  const filteredArticles = useMemo(() => {
    return articles.filter(article => 
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.content.toLowerCase().includes(searchQuery.toLowerCase())
    )
  }, [articles, searchQuery])

  const toggleRowExpansion = (id: string) => {
    const newExpanded = new Set(expandedRows)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedRows(newExpanded)
  }

  const generateCodeSnippets = (article: any): CodeSnippet[] => {
    return [
      {
        language: 'curl',
        label: 'cURL',
        code: `curl -X GET "http://localhost:8000/api/v1/content/articles/${article.id}" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"`
      },
      {
        language: 'python',
        label: 'Python',
        code: `import requests

url = "http://localhost:8000/api/v1/content/articles/${article.id}"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
article = response.json()
print(article)`
      },
      {
        language: 'javascript',
        label: 'JavaScript',
        code: `const response = await fetch('http://localhost:8000/api/v1/content/articles/${article.id}', {
  method: 'GET',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  }
});

const article = await response.json();
console.log(article);`
      }
    ]
  }

  const copyCode = async (code: string) => {
    try {
      await copyToClipboard(code)
      toast.success('Code copied to clipboard!')
    } catch (error) {
      toast.error('Failed to copy code')
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neural-900 dark:text-neural-100">
            Data Explorer
          </h1>
          <p className="text-neural-600 dark:text-neural-400 mt-2">
            Explore and analyze your scraped content with interactive tools.
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button className="btn btn-secondary btn-md">
            <Filter className="w-4 h-4 mr-2" />
            Filters
          </button>
          <button className="btn btn-primary btn-md">
            <Download className="w-4 h-4 mr-2" />
            Export
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="card p-6">
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neural-400" />
            <input
              type="text"
              placeholder="Search articles, content, or metadata..."
              className="input pl-10"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <button className="btn btn-primary btn-md">
            Search
          </button>
        </div>

        {/* Quick Filters */}
        <div className="flex items-center space-x-2 mt-4">
          <span className="text-sm text-neural-600 dark:text-neural-400">Quick filters:</span>
          {['Technology', 'Healthcare', 'Environment', 'Business'].map((filter) => (
            <button
              key={filter}
              onClick={() => {
                setSelectedFilters(prev => 
                  prev.includes(filter) 
                    ? prev.filter(f => f !== filter)
                    : [...prev, filter]
                )
              }}
              className={cn(
                'badge text-xs px-2 py-1 cursor-pointer transition-colors',
                selectedFilters.includes(filter)
                  ? 'badge-primary'
                  : 'badge-neutral hover:badge-primary'
              )}
            >
              {filter}
            </button>
          ))}
        </div>
      </div>

      {/* Data Table */}
      <div className="card overflow-hidden">
        <div className="p-6 border-b border-neural-200 dark:border-neural-700">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
              Articles ({filteredArticles.length})
            </h2>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-neural-500 dark:text-neural-400">
                Showing {filteredArticles.length} results
              </span>
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="p-12 text-center">
            <LoadingSpinner size="lg" />
            <p className="text-neural-500 dark:text-neural-400 mt-4">Loading articles...</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="table">
              <thead>
                <tr>
                  <th className="w-8"></th>
                  <th>Title</th>
                  <th>Source</th>
                  <th>Published</th>
                  <th>Sentiment</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredArticles.map((article) => (
                  <React.Fragment key={article.id}>
                    <tr>
                      <td>
                        <button
                          onClick={() => toggleRowExpansion(article.id)}
                          className="p-1 hover:bg-neutral-100 dark:hover:bg-neural-700 rounded"
                        >
                          {expandedRows.has(article.id) ? (
                            <ChevronDown className="w-4 h-4" />
                          ) : (
                            <ChevronRight className="w-4 h-4" />
                          )}
                        </button>
                      </td>
                      <td>
                        <div className="max-w-md">
                          <h3 className="font-medium text-neural-900 dark:text-neural-100 truncate">
                            {article.title}
                          </h3>
                          <p className="text-sm text-neural-500 dark:text-neural-400 truncate">
                            {article.summary}
                          </p>
                        </div>
                      </td>
                      <td>
                        <div className="flex items-center space-x-2">
                          <Globe className="w-4 h-4 text-neural-400" />
                          <span className="text-sm">{article.source_domain}</span>
                        </div>
                      </td>
                      <td>
                        <div className="text-sm">
                          <div>{formatRelativeTime(article.published_at)}</div>
                          <div className="text-neural-400 text-xs">
                            {formatDate(article.published_at)}
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="flex items-center space-x-2">
                          <div
                            className={cn(
                              'w-2 h-2 rounded-full',
                              article.nlp_data.sentiment > 0.2
                                ? 'bg-success-500'
                                : article.nlp_data.sentiment < -0.2
                                ? 'bg-error-500'
                                : 'bg-neural-400'
                            )}
                          />
                          <span className="text-sm">
                            {article.nlp_data.sentiment > 0 ? '+' : ''}
                            {article.nlp_data.sentiment.toFixed(2)}
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => {
                              setSelectedArticle(article)
                              setShowCodeModal(true)
                            }}
                            className="btn btn-ghost btn-sm"
                          >
                            <Code className="w-3 h-3" />
                          </button>
                          <button className="btn btn-ghost btn-sm">
                            <Eye className="w-3 h-3" />
                          </button>
                          <button className="btn btn-ghost btn-sm">
                            <ExternalLink className="w-3 h-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                    
                    {/* Expanded Row */}
                    <AnimatePresence>
                      {expandedRows.has(article.id) && (
                        <motion.tr
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.2 }}
                        >
                          <td colSpan={6} className="bg-neutral-50 dark:bg-neural-800/50">
                            <div className="p-6 space-y-4">
                              {/* Content Preview */}
                              <div>
                                <h4 className="font-medium text-neural-900 dark:text-neural-100 mb-2">
                                  Content Preview
                                </h4>
                                <p className="text-sm text-neural-600 dark:text-neural-400 line-clamp-3">
                                  {article.content}
                                </p>
                              </div>

                              {/* Metadata Grid */}
                              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div>
                                  <h5 className="font-medium text-neural-900 dark:text-neural-100 mb-2">
                                    Entities
                                  </h5>
                                  <div className="flex flex-wrap gap-1">
                                    {article.nlp_data.entities.map((entity: any, idx: number) => (
                                      <span key={idx} className="badge badge-neutral text-xs">
                                        {entity.text}
                                      </span>
                                    ))}
                                  </div>
                                </div>

                                <div>
                                  <h5 className="font-medium text-neural-900 dark:text-neural-100 mb-2">
                                    Categories
                                  </h5>
                                  <div className="flex flex-wrap gap-1">
                                    {article.nlp_data.categories.map((category: string, idx: number) => (
                                      <span key={idx} className="badge badge-primary text-xs">
                                        {category}
                                      </span>
                                    ))}
                                  </div>
                                </div>

                                <div>
                                  <h5 className="font-medium text-neural-900 dark:text-neural-100 mb-2">
                                    Metrics
                                  </h5>
                                  <div className="space-y-1 text-sm">
                                    <div>Significance: {(article.nlp_data.significance * 100).toFixed(0)}%</div>
                                    <div>Words: {article.page_metadata.word_count}</div>
                                    <div>Reading time: {article.page_metadata.reading_time}min</div>
                                  </div>
                                </div>
                              </div>

                              {/* JSON View Toggle */}
                              <details className="group">
                                <summary className="cursor-pointer text-sm font-medium text-synapse-600 dark:text-synapse-400 hover:text-synapse-700 dark:hover:text-synapse-300">
                                  View Raw JSON
                                </summary>
                                <div className="mt-2 p-4 bg-neural-900 dark:bg-neural-950 rounded-lg overflow-auto">
                                  <pre className="text-xs text-neural-100">
                                    {JSON.stringify(article, null, 2)}
                                  </pre>
                                </div>
                              </details>
                            </div>
                          </td>
                        </motion.tr>
                      )}
                    </AnimatePresence>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Code Generation Modal */}
      <AnimatePresence>
        {showCodeModal && selectedArticle && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
            onClick={() => setShowCodeModal(false)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-neural-800 rounded-xl shadow-xl max-w-4xl w-full mx-4 max-h-[80vh] overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6 border-b border-neural-200 dark:border-neural-700">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
                    Code Snippets - {selectedArticle.title}
                  </h3>
                  <button
                    onClick={() => setShowCodeModal(false)}
                    className="btn btn-ghost btn-sm"
                  >
                    ×
                  </button>
                </div>
              </div>

              <div className="p-6 overflow-y-auto max-h-[60vh]">
                <div className="space-y-6">
                  {generateCodeSnippets(selectedArticle).map((snippet) => (
                    <div key={snippet.language} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-neural-900 dark:text-neural-100">
                          {snippet.label}
                        </h4>
                        <button
                          onClick={() => copyCode(snippet.code)}
                          className="btn btn-ghost btn-sm"
                        >
                          <Copy className="w-3 h-3 mr-1" />
                          Copy
                        </button>
                      </div>
                      <div className="bg-neural-900 dark:bg-neural-950 rounded-lg p-4 overflow-x-auto">
                        <pre className="text-sm text-neural-100">
                          <code>{snippet.code}</code>
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}