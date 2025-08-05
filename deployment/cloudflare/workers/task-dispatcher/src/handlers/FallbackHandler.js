/**
 * Fallback Handler - Manages failover to Vercel Edge Functions
 */

import { ErrorHandler } from '../utils/ErrorHandler.js';

export class FallbackHandler {
  constructor(env, ctx, logger) {
    this.env = env;
    this.ctx = ctx;
    this.logger = logger;
  }
  
  /**
   * Forward request to Vercel Edge Function
   */
  async forwardToVercel(request) {
    try {
      const vercelUrl = this.env.VERCEL_FALLBACK_URL;
      
      if (!vercelUrl) {
        return ErrorHandler.serviceUnavailable('Fallback service not configured');
      }
      
      this.logger.info('Forwarding request to Vercel fallback', {
        url: request.url,
        method: request.method
      });
      
      // Clone the request for forwarding
      const forwardedRequest = new Request(request);
      
      // Update the URL to point to Vercel
      const originalUrl = new URL(request.url);
      const vercelRequestUrl = `${vercelUrl}${originalUrl.pathname}${originalUrl.search}`;
      
      // Create new request with Vercel URL
      const vercelRequest = new Request(vercelRequestUrl, {
        method: forwardedRequest.method,
        headers: {
          ...Object.fromEntries(forwardedRequest.headers),
          'X-Forwarded-From': 'cloudflare-worker',
          'X-Original-Host': originalUrl.host,
          'X-Fallback-Reason': 'cloudflare-worker-fallback'
        },
        body: forwardedRequest.body
      });
      
      // Forward to Vercel with timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      try {
        const response = await fetch(vercelRequest, {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Log the fallback usage
        await this.logFallbackUsage('vercel', 'success', {
          status: response.status,
          url: originalUrl.pathname
        });
        
        this.logger.info('Request forwarded to Vercel successfully', {
          status: response.status,
          url: originalUrl.pathname
        });
        
        // Return the response from Vercel
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: {
            ...Object.fromEntries(response.headers),
            'X-Served-By': 'vercel-fallback',
            'X-Fallback-Used': 'true'
          }
        });
        
      } catch (fetchError) {
        clearTimeout(timeoutId);
        
        if (fetchError.name === 'AbortError') {
          this.logger.error('Vercel fallback timeout', {
            url: originalUrl.pathname,
            timeout: 30000
          });
          
          await this.logFallbackUsage('vercel', 'timeout', {
            url: originalUrl.pathname,
            error: 'Request timeout'
          });
          
          return ErrorHandler.gatewayTimeout('Fallback service timeout');
        }
        
        throw fetchError;
      }
      
    } catch (error) {
      this.logger.error('Error forwarding to Vercel fallback', {
        error: error.message,
        stack: error.stack,
        url: request.url
      });
      
      await this.logFallbackUsage('vercel', 'error', {
        url: new URL(request.url).pathname,
        error: error.message
      });
      
      return ErrorHandler.badGateway('Fallback service unavailable');
    }
  }
  
  /**
   * Get fallback system status
   */
  async getFallbackStatus(request) {
    try {
      const vercelUrl = this.env.VERCEL_FALLBACK_URL;
      
      // Check Vercel availability
      let vercelStatus = 'unknown';
      let vercelResponseTime = null;
      
      if (vercelUrl) {
        try {
          const startTime = Date.now();
          const healthCheckUrl = `${vercelUrl}/health`;
          
          const response = await fetch(healthCheckUrl, {
            method: 'GET',
            signal: AbortSignal.timeout(5000) // 5 second timeout
          });
          
          vercelResponseTime = Date.now() - startTime;
          vercelStatus = response.ok ? 'healthy' : 'unhealthy';
          
        } catch (error) {
          vercelStatus = 'unavailable';
          this.logger.warn('Vercel health check failed', {
            error: error.message
          });
        }
      } else {
        vercelStatus = 'not_configured';
      }
      
      // Get fallback usage statistics
      const stats = await this.getFallbackStats();
      
      const status = {
        fallback_configured: !!vercelUrl,
        vercel: {
          status: vercelStatus,
          url: vercelUrl ? `${vercelUrl}/...` : null,
          response_time_ms: vercelResponseTime
        },
        statistics: stats,
        last_check: new Date().toISOString()
      };
      
      return new Response(JSON.stringify({
        success: true,
        data: status
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error getting fallback status', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to get fallback status');
    }
  }
  
  /**
   * Log fallback usage for monitoring
   */
  async logFallbackUsage(service, result, details = {}) {
    try {
      const logEntry = {
        service,
        result,
        details,
        timestamp: new Date().toISOString(),
        worker_id: this.env.CF_RAY || 'unknown'
      };
      
      // Store in KV with expiration
      const key = `fallback_log:${Date.now()}:${Math.random().toString(36).substr(2, 9)}`;
      await this.env.TASK_STORE.put(
        key,
        JSON.stringify(logEntry),
        { expirationTtl: 86400 * 30 } // 30 days
      );
      
      // Also log to console for immediate visibility
      this.logger.info('Fallback usage logged', logEntry);
      
    } catch (error) {
      this.logger.error('Failed to log fallback usage', {
        error: error.message,
        service,
        result
      });
    }
  }
  
  /**
   * Get fallback usage statistics
   */
  async getFallbackStats() {
    try {
      // This is a simplified version - in production you might want to use
      // Durable Objects or a more sophisticated storage solution for analytics
      
      const stats = {
        total_fallback_requests: 0,
        successful_fallbacks: 0,
        failed_fallbacks: 0,
        timeout_fallbacks: 0,
        last_24h: {
          total: 0,
          successful: 0,
          failed: 0
        },
        services: {
          vercel: {
            total: 0,
            successful: 0,
            failed: 0,
            avg_response_time_ms: null
          }
        }
      };
      
      // In a real implementation, you would:
      // 1. Query KV store for recent fallback logs
      // 2. Aggregate the statistics
      // 3. Calculate averages and trends
      // 4. Return comprehensive metrics
      
      // For now, return basic structure
      return stats;
      
    } catch (error) {
      this.logger.error('Error getting fallback stats', {
        error: error.message
      });
      return null;
    }
  }
  
  /**
   * Test fallback connectivity
   */
  async testFallbackConnectivity() {
    const results = {
      vercel: { available: false, response_time_ms: null, error: null }
    };
    
    // Test Vercel
    if (this.env.VERCEL_FALLBACK_URL) {
      try {
        const startTime = Date.now();
        const response = await fetch(`${this.env.VERCEL_FALLBACK_URL}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000)
        });
        
        results.vercel.response_time_ms = Date.now() - startTime;
        results.vercel.available = response.ok;
        
        if (!response.ok) {
          results.vercel.error = `HTTP ${response.status}`;
        }
        
      } catch (error) {
        results.vercel.error = error.message;
      }
    } else {
      results.vercel.error = 'Not configured';
    }
    
    return results;
  }
  
  /**
   * Determine if fallback should be used
   */
  shouldUseFallback(error, retryCount = 0) {
    // Use fallback for:
    // 1. Network errors
    // 2. 5xx server errors
    // 3. Timeouts
    // 4. After multiple retries
    
    if (retryCount >= 2) {
      return true;
    }
    
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      return true;
    }
    
    if (error.message && error.message.includes('network')) {
      return true;
    }
    
    return false;
  }
}