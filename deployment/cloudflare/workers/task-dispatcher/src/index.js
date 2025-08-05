/**
 * Synapse Task Dispatcher - Cloudflare Worker
 * 
 * This worker handles:
 * - Task submission and triggering
 * - Webhook endpoints for external task submission
 * - Health checks and monitoring
 * - Failover to Vercel Edge Functions
 */

import { TaskCoordinator } from './durable-objects/TaskCoordinator.js';
import { TaskHandler } from './handlers/TaskHandler.js';
import { WebhookHandler } from './handlers/WebhookHandler.js';
import { HealthHandler } from './handlers/HealthHandler.js';
import { FallbackHandler } from './handlers/FallbackHandler.js';
import { AuthMiddleware } from './middleware/AuthMiddleware.js';
import { RateLimitMiddleware } from './middleware/RateLimitMiddleware.js';
import { ErrorHandler } from './utils/ErrorHandler.js';
import { Logger } from './utils/Logger.js';

// Export Durable Object class
export { TaskCoordinator };

/**
 * Main worker fetch handler
 */
export default {
  async fetch(request, env, ctx) {
    const logger = new Logger(env.ENVIRONMENT || 'development');
    
    try {
      // Parse request
      const url = new URL(request.url);
      const path = url.pathname;
      const method = request.method;
      
      logger.info('Request received', {
        method,
        path,
        userAgent: request.headers.get('User-Agent'),
        timestamp: new Date().toISOString()
      });
      
      // CORS handling
      if (method === 'OPTIONS') {
        return new Response(null, {
          status: 204,
          headers: {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key, X-Webhook-Signature',
            'Access-Control-Max-Age': '86400'
          }
        });
      }
      
      // Initialize handlers
      const taskHandler = new TaskHandler(env, ctx, logger);
      const webhookHandler = new WebhookHandler(env, ctx, logger);
      const healthHandler = new HealthHandler(env, ctx, logger);
      const fallbackHandler = new FallbackHandler(env, ctx, logger);
      
      // Route handling
      const router = {
        // Health and status endpoints
        'GET /health': () => healthHandler.health(request),
        'GET /status': () => healthHandler.status(request),
        'GET /metrics': () => healthHandler.metrics(request),
        
        // Task management endpoints
        'POST /tasks': () => taskHandler.submitTask(request),
        'GET /tasks/:id': () => taskHandler.getTaskStatus(request),
        'DELETE /tasks/:id': () => taskHandler.cancelTask(request),
        'GET /tasks': () => taskHandler.listTasks(request),
        
        // Webhook endpoints
        'POST /webhooks/github': () => webhookHandler.handleGitHub(request),
        'POST /webhooks/generic': () => webhookHandler.handleGeneric(request),
        'POST /webhooks/feed-update': () => webhookHandler.handleFeedUpdate(request),
        
        // Cron trigger endpoint
        'POST /cron/process-tasks': () => taskHandler.processCronTasks(request),
        
        // Fallback endpoints
        'POST /fallback/vercel': () => fallbackHandler.forwardToVercel(request),
        'GET /fallback/status': () => fallbackHandler.getFallbackStatus(request)
      };
      
      // Find matching route
      const routeKey = `${method} ${path}`;
      let handler = router[routeKey];
      
      // Handle parameterized routes
      if (!handler) {
        for (const [route, routeHandler] of Object.entries(router)) {
          const [routeMethod, routePath] = route.split(' ');
          if (routeMethod === method && matchRoute(routePath, path)) {
            handler = routeHandler;
            break;
          }
        }
      }
      
      if (!handler) {
        return ErrorHandler.notFound('Endpoint not found');
      }
      
      // Apply middleware
      const authMiddleware = new AuthMiddleware(env, logger);
      const rateLimitMiddleware = new RateLimitMiddleware(env, logger);
      
      // Skip auth for health endpoints
      if (!path.startsWith('/health') && !path.startsWith('/status')) {
        const authResult = await authMiddleware.authenticate(request);
        if (!authResult.success) {
          return ErrorHandler.unauthorized(authResult.error);
        }
        
        // Apply rate limiting
        const rateLimitResult = await rateLimitMiddleware.checkLimit(request);
        if (!rateLimitResult.allowed) {
          return ErrorHandler.rateLimited(rateLimitResult.message);
        }
      }
      
      // Execute handler
      const response = await handler();
      
      // Add CORS headers to response
      const corsHeaders = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key'
      };
      
      // Clone response to add headers
      const newResponse = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: { ...Object.fromEntries(response.headers), ...corsHeaders }
      });
      
      logger.info('Request completed', {
        method,
        path,
        status: response.status,
        duration: Date.now() - new Date(request.headers.get('X-Request-Start') || Date.now())
      });
      
      return newResponse;
      
    } catch (error) {
      logger.error('Unhandled error in worker', {
        error: error.message,
        stack: error.stack,
        url: request.url,
        method: request.method
      });
      
      return ErrorHandler.internalError('Internal server error');
    }
  },
  
  /**
   * Scheduled event handler for cron triggers
   */
  async scheduled(event, env, ctx) {
    const logger = new Logger(env.ENVIRONMENT || 'development');
    
    try {
      logger.info('Scheduled event triggered', {
        cron: event.cron,
        scheduledTime: new Date(event.scheduledTime).toISOString()
      });
      
      const taskHandler = new TaskHandler(env, ctx, logger);
      await taskHandler.processScheduledTasks();
      
      logger.info('Scheduled event completed successfully');
      
    } catch (error) {
      logger.error('Error in scheduled event', {
        error: error.message,
        stack: error.stack,
        cron: event.cron
      });
    }
  }
};

/**
 * Match route patterns with parameters
 */
function matchRoute(pattern, path) {
  const patternParts = pattern.split('/');
  const pathParts = path.split('/');
  
  if (patternParts.length !== pathParts.length) {
    return false;
  }
  
  for (let i = 0; i < patternParts.length; i++) {
    const patternPart = patternParts[i];
    const pathPart = pathParts[i];
    
    if (patternPart.startsWith(':')) {
      // Parameter - matches any value
      continue;
    }
    
    if (patternPart !== pathPart) {
      return false;
    }
  }
  
  return true;
}

/**
 * Extract route parameters
 */
function extractParams(pattern, path) {
  const patternParts = pattern.split('/');
  const pathParts = path.split('/');
  const params = {};
  
  for (let i = 0; i < patternParts.length; i++) {
    const patternPart = patternParts[i];
    const pathPart = pathParts[i];
    
    if (patternPart.startsWith(':')) {
      const paramName = patternPart.slice(1);
      params[paramName] = pathPart;
    }
  }
  
  return params;
}