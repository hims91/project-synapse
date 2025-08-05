/**
 * Feed Coordinator - Durable Object
 * Manages feed configurations, scheduling, and coordination across workers
 */

export class FeedCoordinator {
  constructor(state, env) {
    this.state = state;
    this.env = env;
    this.feeds = new Map();
    this.metrics = new Map();
    this.lastPollTimes = new Map();
    
    // Initialize from storage
    this.initializeFromStorage();
  }

  async initializeFromStorage() {
    try {
      const feedsData = await this.state.storage.get('feeds');
      if (feedsData) {
        this.feeds = new Map(Object.entries(feedsData));
      }
      
      const metricsData = await this.state.storage.get('metrics');
      if (metricsData) {
        this.metrics = new Map(Object.entries(metricsData));
      }
      
      const pollTimesData = await this.state.storage.get('lastPollTimes');
      if (pollTimesData) {
        this.lastPollTimes = new Map(Object.entries(pollTimesData));
      }
      
    } catch (error) {
      console.error('Error initializing from storage:', error);
    }
  }

  async saveToStorage() {
    try {
      await this.state.storage.put('feeds', Object.fromEntries(this.feeds));
      await this.state.storage.put('metrics', Object.fromEntries(this.metrics));
      await this.state.storage.put('lastPollTimes', Object.fromEntries(this.lastPollTimes));
    } catch (error) {
      console.error('Error saving to storage:', error);
    }
  }

  async fetch(request) {
    const url = new URL(request.url);
    const path = url.pathname;
    
    try {
      if (path === '/status') {
        return this.handleStatus();
      }
      
      if (path === '/feeds' && request.method === 'GET') {
        return this.handleListFeeds(url.searchParams);
      }
      
      if (path === '/feeds' && request.method === 'POST') {
        return this.handleAddFeed(request);
      }
      
      if (path.startsWith('/feeds/') && request.method === 'DELETE') {
        const feedId = path.split('/')[2];
        return this.handleRemoveFeed(feedId);
      }
      
      if (path === '/feeds/due') {
        return this.handleGetDueFeeds(url.searchParams);
      }
      
      if (path === '/feeds/batch' && request.method === 'POST') {
        return this.handleGetFeedsBatch(request);
      }
      
      if (path === '/results' && request.method === 'POST') {
        return this.handleUpdateResults(request);
      }
      
      return new Response('Not Found', { status: 404 });
      
    } catch (error) {
      console.error('Coordinator error:', error);
      return new Response(
        JSON.stringify({ error: error.message }),
        { 
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
  }

  async handleStatus() {
    const status = {
      totalFeeds: this.feeds.size,
      enabledFeeds: Array.from(this.feeds.values()).filter(f => f.enabled).length,
      lastUpdate: new Date().toISOString(),
      priorityDistribution: this.getPriorityDistribution(),
      categoryDistribution: this.getCategoryDistribution()
    };
    
    return new Response(JSON.stringify(status), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async handleListFeeds(searchParams) {
    const priority = searchParams.get('priority');
    const category = searchParams.get('category');
    const enabled = searchParams.get('enabled');
    
    let feeds = Array.from(this.feeds.values());
    
    // Apply filters
    if (priority) {
      feeds = feeds.filter(f => f.priority === priority);
    }
    
    if (category) {
      feeds = feeds.filter(f => f.category === category);
    }
    
    if (enabled !== null) {
      const isEnabled = enabled === 'true';
      feeds = feeds.filter(f => f.enabled === isEnabled);
    }
    
    return new Response(JSON.stringify({ feeds }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async handleAddFeed(request) {
    const feedConfig = await request.json();
    
    // Validate required fields
    if (!feedConfig.url || !feedConfig.name) {
      return new Response(
        JSON.stringify({ error: 'Feed URL and name are required' }),
        { 
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
    
    // Generate feed ID if not provided
    if (!feedConfig.feedId) {
      feedConfig.feedId = this.generateFeedId(feedConfig.url);
    }
    
    // Set defaults
    const feed = {
      feedId: feedConfig.feedId,
      url: feedConfig.url,
      name: feedConfig.name,
      category: feedConfig.category || 'general',
      priority: feedConfig.priority || 'normal',
      enabled: feedConfig.enabled !== false,
      customInterval: feedConfig.customInterval || null,
      userAgent: feedConfig.userAgent || 'Project-Synapse-Dendrite/1.0',
      timeout: feedConfig.timeout || 30,
      maxRetries: feedConfig.maxRetries || 3,
      tags: feedConfig.tags || [],
      metadata: feedConfig.metadata || {},
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    
    // Store feed
    this.feeds.set(feed.feedId, feed);
    
    // Initialize metrics
    this.metrics.set(feed.feedId, {
      totalPolls: 0,
      successfulPolls: 0,
      failedPolls: 0,
      lastPollTime: null,
      lastSuccessTime: null,
      lastNewItems: 0,
      averageItemsPerPoll: 0,
      consecutiveFailures: 0,
      consecutiveEmptyPolls: 0
    });
    
    await this.saveToStorage();
    
    return new Response(JSON.stringify({ 
      message: 'Feed added successfully',
      feed: feed
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async handleRemoveFeed(feedId) {
    if (!this.feeds.has(feedId)) {
      return new Response(
        JSON.stringify({ error: 'Feed not found' }),
        { 
          status: 404,
          headers: { 'Content-Type': 'application/json' }
        }
      );
    }
    
    this.feeds.delete(feedId);
    this.metrics.delete(feedId);
    this.lastPollTimes.delete(feedId);
    
    await this.saveToStorage();
    
    return new Response(JSON.stringify({ 
      message: 'Feed removed successfully'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async handleGetDueFeeds(searchParams) {
    const priority = searchParams.get('priority');
    const now = Date.now();
    
    const dueFeeds = [];
    
    for (const [feedId, feed] of this.feeds) {
      if (!feed.enabled) continue;
      
      // Filter by priority if specified
      if (priority && feed.priority !== priority) continue;
      
      // Check if feed is due for polling
      const lastPollTime = this.lastPollTimes.get(feedId);
      const interval = this.getPollingInterval(feed);
      
      if (!lastPollTime || (now - new Date(lastPollTime).getTime()) >= interval * 1000) {
        dueFeeds.push(feed);
      }
    }
    
    // Sort by priority
    dueFeeds.sort((a, b) => this.getPriorityWeight(a.priority) - this.getPriorityWeight(b.priority));
    
    return new Response(JSON.stringify({ feeds: dueFeeds }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async handleGetFeedsBatch(request) {
    const { feedIds } = await request.json();
    
    const feeds = feedIds
      .map(id => this.feeds.get(id))
      .filter(Boolean);
    
    return new Response(JSON.stringify({ feeds }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  async handleUpdateResults(request) {
    const { results } = await request.json();
    
    for (const result of results) {
      this.updateFeedMetrics(result);
    }
    
    await this.saveToStorage();
    
    return new Response(JSON.stringify({ 
      message: 'Results updated successfully'
    }), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  updateFeedMetrics(result) {
    const feedId = result.feedId;
    const metrics = this.metrics.get(feedId);
    
    if (!metrics) return;
    
    metrics.totalPolls++;
    metrics.lastPollTime = result.timestamp;
    
    if (result.success) {
      metrics.successfulPolls++;
      metrics.lastSuccessTime = result.timestamp;
      metrics.consecutiveFailures = 0;
      
      const newItems = result.newItems || 0;
      metrics.lastNewItems = newItems;
      
      if (newItems === 0) {
        metrics.consecutiveEmptyPolls++;
      } else {
        metrics.consecutiveEmptyPolls = 0;
      }
      
      // Update average items per poll
      const totalItems = (metrics.averageItemsPerPoll * (metrics.successfulPolls - 1)) + newItems;
      metrics.averageItemsPerPoll = totalItems / metrics.successfulPolls;
      
    } else {
      metrics.failedPolls++;
      metrics.consecutiveFailures++;
    }
    
    // Auto-adjust priority based on metrics
    this.autoAdjustPriority(feedId);
    
    // Update last poll time
    this.lastPollTimes.set(feedId, result.timestamp);
  }

  autoAdjustPriority(feedId) {
    const feed = this.feeds.get(feedId);
    const metrics = this.metrics.get(feedId);
    
    if (!feed || !metrics || feed.customInterval) return;
    
    const oldPriority = feed.priority;
    let newPriority = oldPriority;
    
    const successRate = metrics.totalPolls > 0 ? 
      (metrics.successfulPolls / metrics.totalPolls) * 100 : 0;
    
    // Promote to higher priority if very active
    if (metrics.averageItemsPerPoll > 10 && successRate > 90) {
      if (oldPriority === 'normal') {
        newPriority = 'high';
      } else if (oldPriority === 'low') {
        newPriority = 'normal';
      }
    }
    // Demote if consistently inactive or unhealthy
    else if (metrics.consecutiveEmptyPolls > 20 || successRate < 70 || metrics.consecutiveFailures > 5) {
      if (oldPriority === 'high') {
        newPriority = 'normal';
      } else if (oldPriority === 'normal') {
        newPriority = 'low';
      } else if (oldPriority === 'low') {
        newPriority = 'inactive';
      }
    }
    
    if (newPriority !== oldPriority) {
      feed.priority = newPriority;
      feed.updatedAt = new Date().toISOString();
      
      console.log(`Auto-adjusted feed priority: ${feedId} ${oldPriority} -> ${newPriority}`);
    }
  }

  getPollingInterval(feed) {
    // Priority-based intervals (in seconds)
    const intervals = {
      critical: 300,   // 5 minutes
      high: 900,       // 15 minutes
      normal: 1800,    // 30 minutes
      low: 3600,       // 1 hour
      inactive: 14400  // 4 hours
    };
    
    if (feed.customInterval) {
      return feed.customInterval * 60; // Convert minutes to seconds
    }
    
    let baseInterval = intervals[feed.priority] || intervals.normal;
    
    // Adaptive adjustment based on metrics
    const metrics = this.metrics.get(feed.feedId);
    if (metrics) {
      const successRate = metrics.totalPolls > 0 ? 
        (metrics.successfulPolls / metrics.totalPolls) * 100 : 0;
      
      // Increase interval for unhealthy feeds
      if (successRate < 70 || metrics.consecutiveFailures > 5) {
        baseInterval *= 2;
      }
      // Decrease interval for very active feeds
      else if (metrics.averageItemsPerPoll > 5) {
        baseInterval = Math.max(baseInterval / 2, 300); // Minimum 5 minutes
      }
    }
    
    return baseInterval;
  }

  getPriorityWeight(priority) {
    const weights = {
      critical: 0,
      high: 1,
      normal: 2,
      low: 3,
      inactive: 4
    };
    return weights[priority] || 2;
  }

  getPriorityDistribution() {
    const distribution = {};
    for (const feed of this.feeds.values()) {
      distribution[feed.priority] = (distribution[feed.priority] || 0) + 1;
    }
    return distribution;
  }

  getCategoryDistribution() {
    const distribution = {};
    for (const feed of this.feeds.values()) {
      distribution[feed.category] = (distribution[feed.category] || 0) + 1;
    }
    return distribution;
  }

  generateFeedId(url) {
    // Simple hash function for feed ID generation
    let hash = 0;
    for (let i = 0; i < url.length; i++) {
      const char = url.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).substring(0, 16);
  }
}