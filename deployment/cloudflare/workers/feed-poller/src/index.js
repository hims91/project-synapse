/**
 * Project Synapse - Feed Poller Worker
 * Cloudflare Workers implementation of the Dendrites layer
 * 
 * This worker handles:
 * - Scheduled feed polling based on priority
 * - Feed parsing with Google News URL decoding
 * - Metrics collection and storage
 * - Feed coordination across multiple workers
 */

import { FeedParser } from './feedParser.js';
import { FeedCoordinator } from './feedCoordinator.js';
import { MetricsCollector } from './metricsCollector.js';
import { GoogleNewsDecoder } from './googleNewsDecoder.js';

// Worker configuration
const CONFIG = {
  MAX_CONCURRENT_POLLS: 10,
  DEFAULT_TIMEOUT: 30000,
  BATCH_SIZE: 5,
  USER_AGENT: 'Project-Synapse-Dendrite/1.0',
  
  // Priority-based polling intervals (in seconds)
  POLL_INTERVALS: {
    critical: 300,   // 5 minutes
    high: 900,       // 15 minutes
    normal: 1800,    // 30 minutes
    low: 3600,       // 1 hour
    inactive: 14400  // 4 hours
  }
};

/**
 * Main worker entry point
 */
export default {
  /**
   * Handle HTTP requests
   */
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    
    try {
      // Route handling
      if (path === '/health') {
        return handleHealthCheck(env);
      }
      
      if (path === '/status') {
        return handleStatus(env);
      }
      
      if (path === '/feeds' && request.method === 'GET') {
        return handleListFeeds(env);
      }
      
      if (path === '/feeds' && request.method === 'POST') {
        return handleAddFeed(request, env);
      }
      
      if (path.startsWith('/feeds/') && request.method === 'DELETE') {
        const feedId = path.split('/')[2];
        return handleRemoveFeed(feedId, env);
      }
      
      if (path === '/poll' && request.method === 'POST') {
        return handleManualPoll(request, env, ctx);
      }
      
      if (path === '/metrics') {
        return handleMetrics(env);
      }
      
      return new Response('Not Found', { status: 404 });
      
    } catch (error) {
      console.error('Worker error:', error);
      return new Response(
        JSON.stringify({ error: error.message }),
        { 
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
  },

  /**
   * Handle scheduled events (cron triggers)
   */
  async scheduled(event, env, ctx) {
    console.log('Scheduled event triggered:', event.cron);
    
    try {
      // Determine which feeds to poll based on cron schedule
      const feedsToPoll = await getFeedsForSchedule(event.cron, env);
      
      if (feedsToPoll.length === 0) {
        console.log('No feeds to poll for this schedule');
        return;
      }
      
      console.log(`Polling ${feedsToPoll.length} feeds for schedule: ${event.cron}`);
      
      // Execute polling in batches
      const results = await pollFeedsInBatches(feedsToPoll, env, ctx);
      
      // Store results and update metrics
      await storePollingResults(results, env);
      
      console.log(`Completed polling cycle: ${results.length} feeds processed`);
      
    } catch (error) {
      console.error('Scheduled polling error:', error);
    }
  }
};

/**
 * Health check endpoint
 */
async function handleHealthCheck(env) {
  const health = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    environment: env.ENVIRONMENT || 'development'
  };
  
  return new Response(JSON.stringify(health), {
    headers: { 'Content-Type': 'application/json' }
  });
}

/**
 * Status endpoint with detailed information
 */
async function handleStatus(env) {
  try {
    // Get feed coordinator status
    const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
    const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
    const coordinatorStatus = await coordinator.fetch(new Request('http://coordinator/status'));
    const coordinatorData = await coordinatorStatus.json();
    
    // Get metrics
    const metrics = await getOverallMetrics(env);
    
    const status = {
      timestamp: new Date().toISOString(),
      coordinator: coordinatorData,
      metrics: metrics,
      config: {
        maxConcurrentPolls: CONFIG.MAX_CONCURRENT_POLLS,
        batchSize: CONFIG.BATCH_SIZE,
        pollIntervals: CONFIG.POLL_INTERVALS
      }
    };
    
    return new Response(JSON.stringify(status, null, 2), {
      headers: { 'Content-Type': 'application/json' }
    });
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Failed to get status', details: error.message }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * List all feeds
 */
async function handleListFeeds(env) {
  try {
    const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
    const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
    const response = await coordinator.fetch(new Request('http://coordinator/feeds'));
    
    return response;
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Failed to list feeds', details: error.message }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Add a new feed
 */
async function handleAddFeed(request, env) {
  try {
    const feedConfig = await request.json();
    
    // Validate feed configuration
    if (!feedConfig.url || !feedConfig.name) {
      return new Response(
        JSON.stringify({ error: 'Feed URL and name are required' }),
        { 
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
    
    // Add feed via coordinator
    const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
    const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
    
    const addRequest = new Request('http://coordinator/feeds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(feedConfig)
    });
    
    const response = await coordinator.fetch(addRequest);
    return response;
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Failed to add feed', details: error.message }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Remove a feed
 */
async function handleRemoveFeed(feedId, env) {
  try {
    const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
    const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
    
    const deleteRequest = new Request(`http://coordinator/feeds/${feedId}`, {
      method: 'DELETE'
    });
    
    const response = await coordinator.fetch(deleteRequest);
    return response;
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Failed to remove feed', details: error.message }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Manual poll trigger
 */
async function handleManualPoll(request, env, ctx) {
  try {
    const { feedIds, priority } = await request.json();
    
    // Get feeds to poll
    let feedsToPoll;
    if (feedIds && feedIds.length > 0) {
      feedsToPoll = await getFeedsByIds(feedIds, env);
    } else if (priority) {
      feedsToPoll = await getFeedsByPriority(priority, env);
    } else {
      feedsToPoll = await getAllFeeds(env);
    }
    
    if (feedsToPoll.length === 0) {
      return new Response(
        JSON.stringify({ message: 'No feeds to poll' }),
        { headers: { 'Content-Type': 'application/json' } }
      );
    }
    
    // Execute polling
    const results = await pollFeedsInBatches(feedsToPoll, env, ctx);
    
    // Store results
    await storePollingResults(results, env);
    
    return new Response(JSON.stringify({
      message: 'Manual polling completed',
      feedsPolled: results.length,
      results: results
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Manual polling failed', details: error.message }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Get metrics endpoint
 */
async function handleMetrics(env) {
  try {
    const metrics = await getOverallMetrics(env);
    
    return new Response(JSON.stringify(metrics, null, 2), {
      headers: { 'Content-Type': 'application/json' }
    });
    
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Failed to get metrics', details: error.message }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Get feeds that should be polled for a given cron schedule
 */
async function getFeedsForSchedule(cronExpression, env) {
  try {
    const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
    const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
    
    // Map cron expressions to priorities
    let priority;
    if (cronExpression === '*/5 * * * *') {
      priority = 'critical';
    } else if (cronExpression === '*/15 * * * *') {
      priority = 'high';
    } else if (cronExpression === '*/30 * * * *') {
      priority = 'normal';
    } else if (cronExpression === '0 * * * *') {
      priority = 'low';
    } else {
      priority = 'normal'; // Default
    }
    
    const response = await coordinator.fetch(
      new Request(`http://coordinator/feeds/due?priority=${priority}`)
    );
    
    if (response.ok) {
      const data = await response.json();
      return data.feeds || [];
    }
    
    return [];
    
  } catch (error) {
    console.error('Error getting feeds for schedule:', error);
    return [];
  }
}

/**
 * Poll feeds in batches to avoid overwhelming servers
 */
async function pollFeedsInBatches(feeds, env, ctx) {
  const results = [];
  const batchSize = CONFIG.BATCH_SIZE;
  
  for (let i = 0; i < feeds.length; i += batchSize) {
    const batch = feeds.slice(i, i + batchSize);
    
    console.log(`Polling batch ${Math.floor(i / batchSize) + 1}: ${batch.length} feeds`);
    
    // Poll batch concurrently
    const batchPromises = batch.map(feed => pollSingleFeed(feed, env));
    const batchResults = await Promise.allSettled(batchPromises);
    
    // Process results
    for (let j = 0; j < batchResults.length; j++) {
      const result = batchResults[j];
      if (result.status === 'fulfilled') {
        results.push(result.value);
      } else {
        // Create error result
        results.push({
          feedId: batch[j].feedId,
          success: false,
          error: result.reason.message,
          timestamp: new Date().toISOString()
        });
      }
    }
    
    // Small delay between batches
    if (i + batchSize < feeds.length) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  return results;
}

/**
 * Poll a single feed
 */
async function pollSingleFeed(feedConfig, env) {
  const startTime = Date.now();
  const timestamp = new Date().toISOString();
  
  console.log(`Polling feed: ${feedConfig.name} (${feedConfig.url})`);
  
  try {
    // Fetch feed content
    const response = await fetch(feedConfig.url, {
      headers: {
        'User-Agent': CONFIG.USER_AGENT,
        'Accept': 'application/rss+xml, application/xml, text/xml'
      },
      cf: {
        cacheTtl: 300, // Cache for 5 minutes
        cacheEverything: true
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const content = await response.text();
    
    // Parse feed
    const parser = new FeedParser();
    const parsedFeed = await parser.parseFeed(content, feedConfig.url);
    
    // Process items and decode Google News URLs
    const processedItems = [];
    const decoder = new GoogleNewsDecoder();
    
    for (const item of parsedFeed.items) {
      // Decode Google News URLs
      if (item.link && item.link.includes('news.google.com')) {
        const decodedUrl = decoder.decodeUrl(item.link);
        if (decodedUrl) {
          item.link = decodedUrl;
          item.googleNewsDecoded = true;
        }
      }
      
      processedItems.push(item);
    }
    
    // Filter new items (check against cache)
    const newItems = await filterNewItems(feedConfig.feedId, processedItems, env);
    
    const responseTime = Date.now() - startTime;
    
    const result = {
      feedId: feedConfig.feedId,
      feedName: feedConfig.name,
      success: true,
      timestamp: timestamp,
      newItems: newItems.length,
      totalItems: processedItems.length,
      responseTime: responseTime,
      items: newItems
    };
    
    console.log(`Feed poll successful: ${feedConfig.name} - ${newItems.length} new items`);
    
    return result;
    
  } catch (error) {
    const responseTime = Date.now() - startTime;
    
    const result = {
      feedId: feedConfig.feedId,
      feedName: feedConfig.name,
      success: false,
      timestamp: timestamp,
      error: error.message,
      responseTime: responseTime
    };
    
    console.error(`Feed poll failed: ${feedConfig.name} - ${error.message}`);
    
    return result;
  }
}

/**
 * Filter new items by checking against cached items
 */
async function filterNewItems(feedId, items, env) {
  try {
    // Get cached item IDs
    const cacheKey = `feed_items:${feedId}`;
    const cachedItemsJson = await env.FEED_CACHE.get(cacheKey);
    const cachedItems = cachedItemsJson ? JSON.parse(cachedItemsJson) : [];
    const cachedItemIds = new Set(cachedItems.map(item => item.guid || item.link));
    
    // Filter new items
    const newItems = items.filter(item => {
      const itemId = item.guid || item.link;
      return itemId && !cachedItemIds.has(itemId);
    });
    
    // Update cache with new items (keep last 100 items)
    const updatedItems = [...newItems, ...cachedItems].slice(0, 100);
    await env.FEED_CACHE.put(cacheKey, JSON.stringify(updatedItems), {
      expirationTtl: 86400 * 7 // 7 days
    });
    
    return newItems;
    
  } catch (error) {
    console.error('Error filtering new items:', error);
    return items; // Return all items if filtering fails
  }
}

/**
 * Store polling results and update metrics
 */
async function storePollingResults(results, env) {
  try {
    // Update coordinator with results
    const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
    const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
    
    const updateRequest = new Request('http://coordinator/results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ results })
    });
    
    await coordinator.fetch(updateRequest);
    
    // Store metrics
    const metricsCollector = new MetricsCollector(env.FEED_METRICS);
    await metricsCollector.recordPollingResults(results);
    
  } catch (error) {
    console.error('Error storing polling results:', error);
  }
}

/**
 * Get overall metrics
 */
async function getOverallMetrics(env) {
  try {
    const metricsCollector = new MetricsCollector(env.FEED_METRICS);
    return await metricsCollector.getOverallMetrics();
  } catch (error) {
    console.error('Error getting metrics:', error);
    return { error: 'Failed to get metrics' };
  }
}

/**
 * Helper functions for feed retrieval
 */
async function getFeedsByIds(feedIds, env) {
  const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
  const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
  
  const response = await coordinator.fetch(
    new Request(`http://coordinator/feeds/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feedIds })
    })
  );
  
  if (response.ok) {
    const data = await response.json();
    return data.feeds || [];
  }
  
  return [];
}

async function getFeedsByPriority(priority, env) {
  const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
  const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
  
  const response = await coordinator.fetch(
    new Request(`http://coordinator/feeds?priority=${priority}`)
  );
  
  if (response.ok) {
    const data = await response.json();
    return data.feeds || [];
  }
  
  return [];
}

async function getAllFeeds(env) {
  const coordinatorId = env.FEED_COORDINATOR.idFromName('main');
  const coordinator = env.FEED_COORDINATOR.get(coordinatorId);
  
  const response = await coordinator.fetch(
    new Request('http://coordinator/feeds')
  );
  
  if (response.ok) {
    const data = await response.json();
    return data.feeds || [];
  }
  
  return [];
}