/**
 * Project Synapse - Dendrites (Edge Cache)
 * Cloudflare Worker for edge caching and request routing
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Add CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    // Handle preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // Handle different routes
    switch (url.pathname) {
      case '/':
        return handleRoot(env, corsHeaders);
      case '/health':
        return handleHealth(corsHeaders);
      case '/cache':
        return handleCache(request, env, corsHeaders);
      case '/proxy':
        return handleProxy(request, env, corsHeaders);
      default:
        return new Response('Not Found', {
          status: 404,
          headers: corsHeaders
        });
    }
  }
};

async function handleRoot(env, corsHeaders) {
  return Response.json({
    service: 'Project Synapse - Dendrites',
    component: 'edge-cache',
    version: '1.0.0',
    status: 'operational',
    environment: env.ENVIRONMENT || 'production',
    hub_url: env.SYNAPSE_HUB_URL,
    endpoints: {
      health: '/health',
      cache: '/cache',
      proxy: '/proxy'
    },
    timestamp: new Date().toISOString()
  }, { headers: corsHeaders });
}

async function handleHealth(corsHeaders) {
  return Response.json({
    status: 'healthy',
    component: 'dendrites',
    version: '1.0.0',
    uptime: Date.now(),
    timestamp: new Date().toISOString()
  }, { headers: corsHeaders });
}

async function handleCache(request, env, corsHeaders) {
  if (request.method === 'GET') {
    // Get cached data
    const url = new URL(request.url);
    const key = url.searchParams.get('key');

    if (!key) {
      return Response.json({
        error: 'Key parameter is required'
      }, { status: 400, headers: corsHeaders });
    }

    try {
      const cached = await env.CACHE.get(key);
      return Response.json({
        key,
        data: cached ? JSON.parse(cached) : null,
        cached_at: cached ? new Date().toISOString() : null
      }, { headers: corsHeaders });
    } catch (error) {
      return Response.json({
        error: 'Failed to retrieve cache'
      }, { status: 500, headers: corsHeaders });
    }
  } else if (request.method === 'POST') {
    // Set cached data
    try {
      const { key, data, ttl = 3600 } = await request.json();

      if (!key || !data) {
        return Response.json({
          error: 'Key and data are required'
        }, { status: 400, headers: corsHeaders });
      }

      await env.CACHE.put(key, JSON.stringify(data), { expirationTtl: ttl });

      return Response.json({
        status: 'success',
        message: 'Data cached successfully',
        key,
        ttl
      }, { headers: corsHeaders });
    } catch (error) {
      return Response.json({
        error: 'Failed to cache data'
      }, { status: 500, headers: corsHeaders });
    }
  }

  return new Response('Method not allowed', {
    status: 405,
    headers: corsHeaders
  });
}

async function handleProxy(request, env, corsHeaders) {
  // Proxy requests to the Central Hub with caching
  const url = new URL(request.url);
  const targetPath = url.searchParams.get('path') || '/';
  const hubUrl = env.SYNAPSE_HUB_URL || 'https://synapse-central-hub.onrender.com';

  try {
    // Check cache first
    const cacheKey = `proxy:${targetPath}`;
    const cached = await env.CACHE.get(cacheKey);

    if (cached) {
      const cachedData = JSON.parse(cached);
      return Response.json(cachedData, {
        headers: {
          ...corsHeaders,
          'X-Cache': 'HIT'
        }
      });
    }

    // Fetch from hub
    const response = await fetch(`${hubUrl}${targetPath}`, {
      method: request.method,
      headers: {
        'Authorization': `Bearer ${env.DENDRITES_API_KEY}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      return Response.json({
        error: 'Hub request failed',
        status: response.status
      }, { status: response.status, headers: corsHeaders });
    }

    const data = await response.json();

    // Cache the response for 5 minutes
    await env.CACHE.put(cacheKey, JSON.stringify(data), { expirationTtl: 300 });

    return Response.json(data, {
      headers: {
        ...corsHeaders,
        'X-Cache': 'MISS'
      }
    });

  } catch (error) {
    return Response.json({
      error: 'Proxy request failed',
      message: error.message
    }, { status: 500, headers: corsHeaders });
  }
}

