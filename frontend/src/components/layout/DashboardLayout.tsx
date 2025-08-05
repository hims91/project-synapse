import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Menu, 
  X, 
  Home, 
  Database, 
  Globe, 
  Code, 
  BarChart3, 
  Settings,
  Sun,
  Moon,
  Monitor,
  LogOut,
  User,
  Bell,
  Search
} from 'lucide-react'

import { useAuth } from '../../contexts/AuthContext'
import { useTheme } from '../../contexts/ThemeContext'
import { useLocation, Link } from 'react-router-dom'
import { cn } from '../../lib/utils'

interface DashboardLayoutProps {
  children: React.ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Data Explorer', href: '/explorer', icon: Database },
  { name: 'Scrape', href: '/scrape', icon: Globe },
  { name: 'API Playground', href: '/playground', icon: Code },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Settings', href: '/settings', icon: Settings },
]

const themeOptions = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'system', label: 'System', icon: Monitor },
] as const

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const [themeMenuOpen, setThemeMenuOpen] = useState(false)
  
  const { user, logout } = useAuth()
  const { theme, setTheme, actualTheme } = useTheme()
  const location = useLocation()

  const isCurrentPath = (path: string) => {
    if (path === '/') {
      return location.pathname === '/'
    }
    return location.pathname.startsWith(path)
  }

  return (
    <div className="min-h-screen bg-neutral-50 dark:bg-neural-950">
      {/* Mobile sidebar backdrop */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.div
        initial={false}
        animate={{
          x: sidebarOpen ? 0 : '-100%',
        }}
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-neural-900',
          'border-r border-neural-200 dark:border-neural-700',
          'lg:translate-x-0 lg:static lg:inset-0',
          'transition-transform duration-300 ease-in-out lg:transition-none'
        )}
      >
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-between px-6 border-b border-neural-200 dark:border-neural-700">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-synapse-500 to-synapse-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">S</span>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-neural-900 dark:text-neural-100">
                  Synapse
                </h1>
                <p className="text-xs text-neural-500 dark:text-neural-400">
                  v2.2.0
                </p>
              </div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-1 rounded-md text-neural-500 hover:text-neural-700 dark:text-neural-400 dark:hover:text-neural-200"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon
              const isActive = isCurrentPath(item.href)
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={cn(
                    'flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200',
                    isActive
                      ? 'bg-synapse-100 dark:bg-synapse-900/30 text-synapse-700 dark:text-synapse-300'
                      : 'text-neural-700 dark:text-neural-300 hover:bg-neutral-100 dark:hover:bg-neural-800'
                  )}
                >
                  <Icon className="w-5 h-5 mr-3" />
                  {item.name}
                </Link>
              )
            })}
          </nav>

          {/* User section */}
          <div className="p-4 border-t border-neural-200 dark:border-neural-700">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-synapse-100 dark:bg-synapse-900/30 rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-synapse-600 dark:text-synapse-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-neural-900 dark:text-neural-100 truncate">
                  {user?.name || 'Developer'}
                </p>
                <p className="text-xs text-neural-500 dark:text-neural-400 truncate">
                  {user?.tier || 'Premium'} Plan
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <header className="sticky top-0 z-30 bg-white/80 dark:bg-neural-900/80 backdrop-blur-sm border-b border-neural-200 dark:border-neural-700">
          <div className="flex h-16 items-center justify-between px-6">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-md text-neural-500 hover:text-neural-700 dark:text-neural-400 dark:hover:text-neural-200"
              >
                <Menu className="w-5 h-5" />
              </button>
              
              {/* Search */}
              <div className="hidden md:block">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-neural-400" />
                  <input
                    type="text"
                    placeholder="Search articles, feeds..."
                    className="pl-10 pr-4 py-2 w-64 text-sm bg-neutral-100 dark:bg-neural-800 border-0 rounded-lg focus:ring-2 focus:ring-synapse-500"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Notifications */}
              <button className="p-2 rounded-lg text-neural-500 hover:text-neural-700 dark:text-neural-400 dark:hover:text-neural-200 hover:bg-neutral-100 dark:hover:bg-neural-800">
                <Bell className="w-5 h-5" />
              </button>

              {/* Theme selector */}
              <div className="relative">
                <button
                  onClick={() => setThemeMenuOpen(!themeMenuOpen)}
                  className="p-2 rounded-lg text-neural-500 hover:text-neural-700 dark:text-neural-400 dark:hover:text-neural-200 hover:bg-neutral-100 dark:hover:bg-neural-800"
                >
                  {actualTheme === 'dark' ? (
                    <Moon className="w-5 h-5" />
                  ) : (
                    <Sun className="w-5 h-5" />
                  )}
                </button>

                <AnimatePresence>
                  {themeMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95, y: -10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: -10 }}
                      className="absolute right-0 mt-2 w-48 bg-white dark:bg-neural-800 rounded-lg shadow-lg border border-neural-200 dark:border-neural-700 py-1"
                    >
                      {themeOptions.map((option) => {
                        const Icon = option.icon
                        return (
                          <button
                            key={option.value}
                            onClick={() => {
                              setTheme(option.value)
                              setThemeMenuOpen(false)
                            }}
                            className={cn(
                              'flex items-center w-full px-4 py-2 text-sm text-left hover:bg-neutral-100 dark:hover:bg-neural-700',
                              theme === option.value && 'bg-synapse-50 dark:bg-synapse-900/20 text-synapse-600 dark:text-synapse-400'
                            )}
                          >
                            <Icon className="w-4 h-4 mr-3" />
                            {option.label}
                          </button>
                        )
                      })}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* User menu */}
              <div className="relative">
                <button
                  onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center space-x-2 p-2 rounded-lg hover:bg-neutral-100 dark:hover:bg-neural-800"
                >
                  <div className="w-6 h-6 bg-synapse-100 dark:bg-synapse-900/30 rounded-full flex items-center justify-center">
                    <User className="w-3 h-3 text-synapse-600 dark:text-synapse-400" />
                  </div>
                </button>

                <AnimatePresence>
                  {userMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.95, y: -10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: -10 }}
                      className="absolute right-0 mt-2 w-48 bg-white dark:bg-neural-800 rounded-lg shadow-lg border border-neural-200 dark:border-neural-700 py-1"
                    >
                      <div className="px-4 py-2 border-b border-neural-200 dark:border-neural-700">
                        <p className="text-sm font-medium text-neural-900 dark:text-neural-100">
                          {user?.name || 'Developer'}
                        </p>
                        <p className="text-xs text-neural-500 dark:text-neural-400">
                          {user?.email || 'developer@synapse.dev'}
                        </p>
                      </div>
                      <button
                        onClick={logout}
                        className="flex items-center w-full px-4 py-2 text-sm text-left text-error-600 hover:bg-error-50 dark:hover:bg-error-900/20"
                      >
                        <LogOut className="w-4 h-4 mr-3" />
                        Sign out
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="p-6"
          >
            {children}
          </motion.div>
        </main>
      </div>

      {/* Click outside handlers */}
      {themeMenuOpen && (
        <div
          className="fixed inset-0 z-20"
          onClick={() => setThemeMenuOpen(false)}
        />
      )}
      {userMenuOpen && (
        <div
          className="fixed inset-0 z-20"
          onClick={() => setUserMenuOpen(false)}
        />
      )}
    </div>
  )
}