import React from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'

interface ErrorBoundaryState {
  hasError: boolean
  error?: Error
}

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ComponentType<{ error: Error; resetError: () => void }>
}

export default class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo)
  }

  resetError = () => {
    this.setState({ hasError: false, error: undefined })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback
        return (
          <FallbackComponent
            error={this.state.error!}
            resetError={this.resetError}
          />
        )
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-neutral-50 dark:bg-neural-950">
          <div className="max-w-md w-full mx-4">
            <div className="bg-white dark:bg-neural-800 rounded-xl shadow-lg p-6 text-center">
              <div className="w-16 h-16 bg-error-100 dark:bg-error-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
                <AlertTriangle className="w-8 h-8 text-error-600 dark:text-error-400" />
              </div>
              
              <h2 className="text-xl font-semibold text-neural-900 dark:text-neural-100 mb-2">
                Something went wrong
              </h2>
              
              <p className="text-neural-600 dark:text-neural-400 mb-6">
                We encountered an unexpected error. Please try refreshing the page.
              </p>
              
              <button
                onClick={this.resetError}
                className="btn btn-primary btn-md"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Try again
              </button>
              
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <details className="mt-6 text-left">
                  <summary className="text-sm text-neural-500 cursor-pointer">
                    Error details
                  </summary>
                  <pre className="mt-2 text-xs bg-neural-100 dark:bg-neural-900 p-3 rounded overflow-auto">
                    {this.state.error.stack}
                  </pre>
                </details>
              )}
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}