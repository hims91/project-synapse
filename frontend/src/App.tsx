import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

import { useAuth } from './contexts/AuthContext'
import { useTheme } from './contexts/ThemeContext'

// Layout Components
import DashboardLayout from './components/layout/DashboardLayout'
import AuthLayout from './components/layout/AuthLayout'

// Page Components
import LoginPage from './pages/auth/LoginPage'
import DashboardPage from './pages/DashboardPage'
import DataExplorerPage from './pages/DataExplorerPage'
import ScrapePage from './pages/ScrapePage'
import APIPlaygroundPage from './pages/APIPlaygroundPage'
import AnalyticsPage from './pages/AnalyticsPage'
import SettingsPage from './pages/SettingsPage'

// Loading and Error Components
import LoadingSpinner from './components/ui/LoadingSpinner'
import ErrorBoundary from './components/ui/ErrorBoundary'

function App() {
  const { user, isLoading } = useAuth()
  const { theme } = useTheme()

  // Apply theme class to document
  React.useEffect(() => {
    document.documentElement.className = theme
  }, [theme])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-50 dark:bg-neural-950">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-neutral-50 dark:bg-neural-950 transition-colors duration-200">
        <AnimatePresence mode="wait">
          <Routes>
            {/* Authentication Routes */}
            {!user ? (
              <Route path="/*" element={
                <AuthLayout>
                  <Routes>
                    <Route path="/login" element={<LoginPage />} />
                    <Route path="/*" element={<Navigate to="/login" replace />} />
                  </Routes>
                </AuthLayout>
              } />
            ) : (
              /* Dashboard Routes */
              <Route path="/*" element={
                <DashboardLayout>
                  <Routes>
                    <Route path="/" element={<DashboardPage />} />
                    <Route path="/dashboard" element={<Navigate to="/" replace />} />
                    <Route path="/explorer" element={<DataExplorerPage />} />
                    <Route path="/scrape" element={<ScrapePage />} />
                    <Route path="/playground" element={<APIPlaygroundPage />} />
                    <Route path="/analytics" element={<AnalyticsPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                    <Route path="/*" element={<Navigate to="/" replace />} />
                  </Routes>
                </DashboardLayout>
              } />
            )}
          </Routes>
        </AnimatePresence>
      </div>
    </ErrorBoundary>
  )
}

export default App