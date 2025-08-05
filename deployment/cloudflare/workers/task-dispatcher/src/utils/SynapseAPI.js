/**
 * Synapse API Client - Handles communication with the main Synapse API
 */

export class SynapseAPI {
  constructor(env, logger) {
    this.env = env;
    this.logger = logger;
    this.baseUrl = env.SYNAPSE_API_URL || 'https://api.synapse.example.com';
    this.apiKey = env.SYNAPSE_API_KEY;
  }
  
  /**
   * Submit a task to the main API
   */
  async submitTask(taskData) {
    try {
      const response = await this.makeRequest('POST', '/tasks', taskData);
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          task_id: data.task_id,
          data: data
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        return {
          success: false,
          error: errorData.message || `HTTP ${response.status}`,
          status: response.status
        };
      }
      
    } catch (error) {
      this.logger.error('Error submitting task to Synapse API', {
        error: error.message,
        task_type: taskData.task_type
      });
      
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * Get task status from the main API
   */
  async getTaskStatus(taskId) {
    try {
      const response = await this.makeRequest('GET', `/tasks/${taskId}`);
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: data
        };
      } else if (response.status === 404) {
        return {
          success: false,
          error: 'Task not found',
          status: 404
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        return {
          success: false,
          error: errorData.message || `HTTP ${response.status}`,
          status: response.status
        };
      }
      
    } catch (error) {
      this.logger.error('Error getting task status from Synapse API', {
        error: error.message,
        task_id: taskId
      });
      
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * Cancel a task via the main API
   */
  async cancelTask(taskId) {
    try {
      const response = await this.makeRequest('DELETE', `/tasks/${taskId}`);
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: data
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        return {
          success: false,
          error: errorData.message || `HTTP ${response.status}`,
          status: response.status
        };
      }
      
    } catch (error) {
      this.logger.error('Error cancelling task via Synapse API', {
        error: error.message,
        task_id: taskId
      });
      
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * List tasks from the main API
   */
  async listTasks(params = {}) {
    try {
      const queryParams = new URLSearchParams();
      
      if (params.limit) queryParams.set('limit', params.limit.toString());
      if (params.offset) queryParams.set('offset', params.offset.toString());
      if (params.status) queryParams.set('status', params.status);
      if (params.task_type) queryParams.set('task_type', params.task_type);
      
      const url = `/tasks?${queryParams.toString()}`;
      const response = await this.makeRequest('GET', url);
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: data.data || data,
          total: data.total || 0
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        return {
          success: false,
          error: errorData.message || `HTTP ${response.status}`,
          status: response.status
        };
      }
      
    } catch (error) {
      this.logger.error('Error listing tasks from Synapse API', {
        error: error.message,
        params
      });
      
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * Trigger scheduled processing
   */
  async triggerScheduledProcessing() {
    try {
      const response = await this.makeRequest('POST', '/system/process-scheduled');
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: data
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        return {
          success: false,
          error: errorData.message || `HTTP ${response.status}`,
          status: response.status
        };
      }
      
    } catch (error) {
      this.logger.error('Error triggering scheduled processing', {
        error: error.message
      });
      
      return {
        success: false,
        error: error.message
      };
    }
  }
  
  /**
   * Get system health from the main API
   */
  async getSystemHealth() {
    try {
      const response = await this.makeRequest('GET', '/health');
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          healthy: true,
          data: data
        };
      } else {
        return {
          success: false,
          healthy: false,
          error: `HTTP ${response.status}`
        };
      }
      
    } catch (error) {
      this.logger.error('Error checking system health', {
        error: error.message
      });
      
      return {
        success: false,
        healthy: false,
        error: error.message
      };
    }
  }
  
  /**
   * Make HTTP request to the main API
   */
  async makeRequest(method, path, body = null) {
    const url = `${this.baseUrl}${path}`;
    
    const headers = {
      'Content-Type': 'application/json',
      'User-Agent': 'Synapse-Worker/1.0',
      'X-Worker-ID': this.env.CF_RAY || 'unknown'
    };
    
    // Add API key if available
    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
      headers['X-API-Key'] = this.apiKey;
    }
    
    const requestOptions = {
      method,
      headers,
      signal: AbortSignal.timeout(30000) // 30 second timeout
    };
    
    if (body && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
      requestOptions.body = JSON.stringify(body);
    }
    
    this.logger.debug('Making API request', {
      method,
      url,
      hasBody: !!body
    });
    
    return fetch(url, requestOptions);
  }
  
  /**
   * Check if the main API is available
   */
  async isAvailable() {
    try {
      const health = await this.getSystemHealth();
      return health.healthy;
    } catch (error) {
      return false;
    }
  }
}