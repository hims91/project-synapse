/**
 * Metrics Collector for Feed Polling
 * Collects and stores metrics about feed polling performance
 */

export class MetricsCollector {
  constructor(kvNamespace) {
    this.kv = kvNamespace;
  }

  /**
   * Record polling results and update metrics
   */
  async recordPollingResults(results) {
    const timestamp = new Date().toISOString();
    const date = timestamp.split('T')[0]; // YYYY-MM-DD format
    
    try {
      // Update daily metrics
      await this.updateDailyMetrics(date, results);
      
      // Update hourly metrics
      const hour = new Date().getUTCHours();
      await this.updateHourlyMetrics(date, hour, results);
      
      // Update feed-specific metrics
      for (const result of results) {
        await this.updateFeedMetrics(result.feedId, result);
      }
      
      // Store latest polling cycle results
      await this.kv.put('latest_polling_cycle', JSON.stringify({
        timestamp,
        results: results.map(r => ({
          feedId: r.feedId,
          feedName: r.feedName,
          success: r.success,
          newItems: r.newItems,
          responseTime: r.responseTime,
          error: r.error
        }))
      }), {
        expirationTtl: 86400 // 24 hours
      });
      
    } catch (error) {
      console.error('Error recording polling results:', error);
    }
  }

  /**
   * Update daily metrics
   */
  async updateDailyMetrics(date, results) {
    const key = `daily_metrics:${date}`;
    
    try {
      const existingData = await this.kv.get(key);
      const metrics = existingData ? JSON.parse(existingData) : {
        date,
        totalPolls: 0,
        successfulPolls: 0,
        failedPolls: 0,
        totalNewItems: 0,
        totalResponseTime: 0,
        feedsPolled: new Set(),
        errors: {}
      };
      
      // Update metrics
      for (const result of results) {
        metrics.totalPolls++;
        metrics.feedsPolled.add(result.feedId);
        
        if (result.success) {
          metrics.successfulPolls++;
          metrics.totalNewItems += result.newItems || 0;
        } else {
          metrics.failedPolls++;
          
          // Track errors
          const errorType = this.categorizeError(result.error);
          metrics.errors[errorType] = (metrics.errors[errorType] || 0) + 1;
        }
        
        if (result.responseTime) {
          metrics.totalResponseTime += result.responseTime;
        }
      }
      
      // Convert Set to Array for JSON serialization
      const metricsToStore = {
        ...metrics,
        feedsPolled: Array.from(metrics.feedsPolled),
        averageResponseTime: metrics.totalPolls > 0 ? 
          metrics.totalResponseTime / metrics.totalPolls : 0,
        successRate: metrics.totalPolls > 0 ? 
          (metrics.successfulPolls / metrics.totalPolls) * 100 : 0
      };
      
      await this.kv.put(key, JSON.stringify(metricsToStore), {
        expirationTtl: 86400 * 30 // 30 days
      });
      
    } catch (error) {
      console.error('Error updating daily metrics:', error);
    }
  }

  /**
   * Update hourly metrics
   */
  async updateHourlyMetrics(date, hour, results) {
    const key = `hourly_metrics:${date}:${hour.toString().padStart(2, '0')}`;
    
    try {
      const existingData = await this.kv.get(key);
      const metrics = existingData ? JSON.parse(existingData) : {
        date,
        hour,
        totalPolls: 0,
        successfulPolls: 0,
        failedPolls: 0,
        totalNewItems: 0,
        totalResponseTime: 0
      };
      
      // Update metrics
      for (const result of results) {
        metrics.totalPolls++;
        
        if (result.success) {
          metrics.successfulPolls++;
          metrics.totalNewItems += result.newItems || 0;
        } else {
          metrics.failedPolls++;
        }
        
        if (result.responseTime) {
          metrics.totalResponseTime += result.responseTime;
        }
      }
      
      metrics.averageResponseTime = metrics.totalPolls > 0 ? 
        metrics.totalResponseTime / metrics.totalPolls : 0;
      metrics.successRate = metrics.totalPolls > 0 ? 
        (metrics.successfulPolls / metrics.totalPolls) * 100 : 0;
      
      await this.kv.put(key, JSON.stringify(metrics), {
        expirationTtl: 86400 * 7 // 7 days
      });
      
    } catch (error) {
      console.error('Error updating hourly metrics:', error);
    }
  }

  /**
   * Update feed-specific metrics
   */
  async updateFeedMetrics(feedId, result) {
    const key = `feed_metrics:${feedId}`;
    
    try {
      const existingData = await this.kv.get(key);
      const metrics = existingData ? JSON.parse(existingData) : {
        feedId,
        totalPolls: 0,
        successfulPolls: 0,
        failedPolls: 0,
        totalNewItems: 0,
        totalResponseTime: 0,
        lastPollTime: null,
        lastSuccessTime: null,
        consecutiveFailures: 0,
        consecutiveEmptyPolls: 0,
        recentErrors: []
      };
      
      // Update metrics
      metrics.totalPolls++;
      metrics.lastPollTime = result.timestamp;
      
      if (result.success) {
        metrics.successfulPolls++;
        metrics.lastSuccessTime = result.timestamp;
        metrics.consecutiveFailures = 0;
        
        const newItems = result.newItems || 0;
        metrics.totalNewItems += newItems;
        
        if (newItems === 0) {
          metrics.consecutiveEmptyPolls++;
        } else {
          metrics.consecutiveEmptyPolls = 0;
        }
        
      } else {
        metrics.failedPolls++;
        metrics.consecutiveFailures++;
        
        // Track recent errors (keep last 10)
        metrics.recentErrors.unshift({
          timestamp: result.timestamp,
          error: result.error,
          responseTime: result.responseTime
        });
        metrics.recentErrors = metrics.recentErrors.slice(0, 10);
      }
      
      if (result.responseTime) {
        metrics.totalResponseTime += result.responseTime;
      }
      
      // Calculate derived metrics
      metrics.averageResponseTime = metrics.totalPolls > 0 ? 
        metrics.totalResponseTime / metrics.totalPolls : 0;
      metrics.successRate = metrics.totalPolls > 0 ? 
        (metrics.successfulPolls / metrics.totalPolls) * 100 : 0;
      metrics.averageItemsPerPoll = metrics.successfulPolls > 0 ? 
        metrics.totalNewItems / metrics.successfulPolls : 0;
      
      await this.kv.put(key, JSON.stringify(metrics), {
        expirationTtl: 86400 * 30 // 30 days
      });
      
    } catch (error) {
      console.error('Error updating feed metrics:', error);
    }
  }

  /**
   * Get overall metrics summary
   */
  async getOverallMetrics() {
    try {
      const today = new Date().toISOString().split('T')[0];
      const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
      
      // Get today's and yesterday's metrics
      const [todayData, yesterdayData, latestCycle] = await Promise.all([
        this.kv.get(`daily_metrics:${today}`),
        this.kv.get(`daily_metrics:${yesterday}`),
        this.kv.get('latest_polling_cycle')
      ]);
      
      const todayMetrics = todayData ? JSON.parse(todayData) : null;
      const yesterdayMetrics = yesterdayData ? JSON.parse(yesterdayData) : null;
      const latestCycleData = latestCycle ? JSON.parse(latestCycle) : null;
      
      return {
        today: todayMetrics,
        yesterday: yesterdayMetrics,
        latestCycle: latestCycleData,
        summary: {
          todayPolls: todayMetrics?.totalPolls || 0,
          todaySuccessRate: todayMetrics?.successRate || 0,
          todayNewItems: todayMetrics?.totalNewItems || 0,
          yesterdayPolls: yesterdayMetrics?.totalPolls || 0,
          yesterdaySuccessRate: yesterdayMetrics?.successRate || 0,
          trend: this.calculateTrend(todayMetrics, yesterdayMetrics)
        }
      };
      
    } catch (error) {
      console.error('Error getting overall metrics:', error);
      return { error: 'Failed to retrieve metrics' };
    }
  }

  /**
   * Get metrics for a specific feed
   */
  async getFeedMetrics(feedId) {
    try {
      const data = await this.kv.get(`feed_metrics:${feedId}`);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error('Error getting feed metrics:', error);
      return null;
    }
  }

  /**
   * Get hourly metrics for a specific date
   */
  async getHourlyMetrics(date) {
    try {
      const hourlyData = [];
      
      for (let hour = 0; hour < 24; hour++) {
        const key = `hourly_metrics:${date}:${hour.toString().padStart(2, '0')}`;
        const data = await this.kv.get(key);
        
        if (data) {
          hourlyData.push(JSON.parse(data));
        } else {
          hourlyData.push({
            date,
            hour,
            totalPolls: 0,
            successfulPolls: 0,
            failedPolls: 0,
            totalNewItems: 0,
            averageResponseTime: 0,
            successRate: 0
          });
        }
      }
      
      return hourlyData;
      
    } catch (error) {
      console.error('Error getting hourly metrics:', error);
      return [];
    }
  }

  /**
   * Categorize error types for better tracking
   */
  categorizeError(error) {
    if (!error) return 'unknown';
    
    const errorStr = error.toLowerCase();
    
    if (errorStr.includes('timeout')) return 'timeout';
    if (errorStr.includes('network') || errorStr.includes('connection')) return 'network';
    if (errorStr.includes('404')) return 'not_found';
    if (errorStr.includes('403') || errorStr.includes('401')) return 'auth_error';
    if (errorStr.includes('500') || errorStr.includes('502') || errorStr.includes('503')) return 'server_error';
    if (errorStr.includes('parse') || errorStr.includes('xml')) return 'parse_error';
    
    return 'other';
  }

  /**
   * Calculate trend between two metric periods
   */
  calculateTrend(current, previous) {
    if (!current || !previous) return 'no_data';
    
    const currentPolls = current.totalPolls || 0;
    const previousPolls = previous.totalPolls || 0;
    
    if (previousPolls === 0) return 'no_baseline';
    
    const change = ((currentPolls - previousPolls) / previousPolls) * 100;
    
    if (change > 10) return 'increasing';
    if (change < -10) return 'decreasing';
    return 'stable';
  }

  /**
   * Clean up old metrics (called periodically)
   */
  async cleanupOldMetrics() {
    try {
      const cutoffDate = new Date(Date.now() - (86400000 * 30)); // 30 days ago
      const cutoffDateStr = cutoffDate.toISOString().split('T')[0];
      
      // This would require listing keys, which is not directly available in KV
      // In practice, we rely on TTL for cleanup
      console.log(`Cleanup would remove metrics older than ${cutoffDateStr}`);
      
    } catch (error) {
      console.error('Error cleaning up old metrics:', error);
    }
  }
}