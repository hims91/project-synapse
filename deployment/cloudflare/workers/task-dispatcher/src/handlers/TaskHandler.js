/**
 * Task Handler - Manages task submission, status, and processing
 */

import { ErrorHandler } from '../utils/ErrorHandler.js';
import { SynapseAPI } from '../utils/SynapseAPI.js';

export class TaskHandler {
  constructor(env, ctx, logger) {
    this.env = env;
    this.ctx = ctx;
    this.logger = logger;
    this.synapseAPI = new SynapseAPI(env, logger);
  }
  
  /**
   * Submit a new task
   */
  async submitTask(request) {
    try {
      const body = await request.json();
      
      // Validate required fields
      if (!body.task_type || !body.payload) {
        return ErrorHandler.badRequest('Missing required fields: task_type, payload');
      }
      
      // Validate task type
      const validTaskTypes = [
        'scrape_url',
        'process_feed',
        'analyze_content',
        'generate_summary',
        'extract_entities',
        'monitor_keyword',
        'send_webhook',
        'cleanup_data',
        'health_check'
      ];
      
      if (!validTaskTypes.includes(body.task_type)) {
        return ErrorHandler.badRequest(`Invalid task_type. Must be one of: ${validTaskTypes.join(', ')}`);
      }
      
      // Prepare task data
      const taskData = {
        task_type: body.task_type,
        payload: body.payload,
        priority: body.priority || 3, // Default to normal priority
        max_retries: body.max_retries || 3,
        scheduled_at: body.scheduled_at || new Date().toISOString(),
        source: 'cloudflare_worker',
        worker_id: this.env.CF_RAY || 'unknown',
        submitted_at: new Date().toISOString()
      };
      
      this.logger.info('Submitting task', {
        task_type: taskData.task_type,
        priority: taskData.priority,
        source: taskData.source
      });
      
      // Submit to main Synapse API
      const result = await this.synapseAPI.submitTask(taskData);
      
      if (!result.success) {
        this.logger.error('Failed to submit task to Synapse API', {
          error: result.error,
          task_type: taskData.task_type
        });
        return ErrorHandler.internalError('Failed to submit task');
      }
      
      // Store task reference in KV for tracking
      await this.env.TASK_STORE.put(
        `task:${result.task_id}`,
        JSON.stringify({
          ...taskData,
          task_id: result.task_id,
          status: 'submitted',
          created_at: new Date().toISOString()
        }),
        { expirationTtl: 86400 * 7 } // 7 days
      );
      
      this.logger.info('Task submitted successfully', {
        task_id: result.task_id,
        task_type: taskData.task_type
      });
      
      return new Response(JSON.stringify({
        success: true,
        task_id: result.task_id,
        status: 'submitted',
        message: 'Task submitted successfully'
      }), {
        status: 201,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error submitting task', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to submit task');
    }
  }
  
  /**
   * Get task status
   */
  async getTaskStatus(request) {
    try {
      const url = new URL(request.url);
      const taskId = url.pathname.split('/').pop();
      
      if (!taskId) {
        return ErrorHandler.badRequest('Task ID is required');
      }
      
      this.logger.info('Getting task status', { task_id: taskId });
      
      // Check local KV store first
      const localTask = await this.env.TASK_STORE.get(`task:${taskId}`, 'json');
      
      // Get status from main API
      const apiResult = await this.synapseAPI.getTaskStatus(taskId);
      
      let taskStatus = {
        task_id: taskId,
        status: 'not_found',
        message: 'Task not found'
      };
      
      if (apiResult.success && apiResult.data) {
        taskStatus = {
          ...apiResult.data,
          worker_info: localTask ? {
            source: localTask.source,
            worker_id: localTask.worker_id,
            submitted_at: localTask.submitted_at
          } : null
        };
      } else if (localTask) {
        // Fallback to local data if API is unavailable
        taskStatus = {
          ...localTask,
          status: 'unknown',
          message: 'Status unavailable from main API'
        };
      }
      
      return new Response(JSON.stringify({
        success: true,
        data: taskStatus
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error getting task status', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to get task status');
    }
  }
  
  /**
   * Cancel a task
   */
  async cancelTask(request) {
    try {
      const url = new URL(request.url);
      const taskId = url.pathname.split('/').pop();
      
      if (!taskId) {
        return ErrorHandler.badRequest('Task ID is required');
      }
      
      this.logger.info('Cancelling task', { task_id: taskId });
      
      // Cancel via main API
      const result = await this.synapseAPI.cancelTask(taskId);
      
      if (!result.success) {
        return ErrorHandler.badRequest(result.error || 'Failed to cancel task');
      }
      
      // Update local KV store
      const localTask = await this.env.TASK_STORE.get(`task:${taskId}`, 'json');
      if (localTask) {
        localTask.status = 'cancelled';
        localTask.cancelled_at = new Date().toISOString();
        await this.env.TASK_STORE.put(`task:${taskId}`, JSON.stringify(localTask));
      }
      
      return new Response(JSON.stringify({
        success: true,
        message: 'Task cancelled successfully'
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error cancelling task', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to cancel task');
    }
  }
  
  /**
   * List tasks (with pagination)
   */
  async listTasks(request) {
    try {
      const url = new URL(request.url);
      const limit = parseInt(url.searchParams.get('limit')) || 50;
      const offset = parseInt(url.searchParams.get('offset')) || 0;
      const status = url.searchParams.get('status');
      const task_type = url.searchParams.get('task_type');
      
      this.logger.info('Listing tasks', { limit, offset, status, task_type });
      
      // Get tasks from main API
      const result = await this.synapseAPI.listTasks({
        limit,
        offset,
        status,
        task_type
      });
      
      if (!result.success) {
        return ErrorHandler.internalError('Failed to list tasks');
      }
      
      return new Response(JSON.stringify({
        success: true,
        data: result.data,
        pagination: {
          limit,
          offset,
          total: result.total || 0
        }
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error listing tasks', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to list tasks');
    }
  }
  
  /**
   * Process cron-triggered tasks
   */
  async processCronTasks(request) {
    try {
      this.logger.info('Processing cron tasks');
      
      // Get Durable Object for coordination
      const coordinatorId = this.env.TASK_COORDINATOR.idFromName('main');
      const coordinator = this.env.TASK_COORDINATOR.get(coordinatorId);
      
      // Process scheduled tasks
      const result = await coordinator.fetch(new Request('https://coordinator/process-scheduled', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }));
      
      const data = await result.json();
      
      this.logger.info('Cron tasks processed', {
        processed: data.processed || 0,
        errors: data.errors || 0
      });
      
      return new Response(JSON.stringify({
        success: true,
        message: 'Cron tasks processed',
        data: data
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error processing cron tasks', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to process cron tasks');
    }
  }
  
  /**
   * Process scheduled tasks (called by cron)
   */
  async processScheduledTasks() {
    try {
      this.logger.info('Processing scheduled tasks');
      
      // This would typically:
      // 1. Check for scheduled tasks that are due
      // 2. Submit them to the main API
      // 3. Update their status
      // 4. Clean up old completed tasks
      
      // For now, we'll just trigger the main API's scheduled processing
      const result = await this.synapseAPI.triggerScheduledProcessing();
      
      if (result.success) {
        this.logger.info('Scheduled processing triggered successfully');
      } else {
        this.logger.error('Failed to trigger scheduled processing', {
          error: result.error
        });
      }
      
    } catch (error) {
      this.logger.error('Error in processScheduledTasks', {
        error: error.message,
        stack: error.stack
      });
    }
  }
}