/**
 * Webhook Handler - Processes incoming webhooks and converts them to tasks
 */

import { ErrorHandler } from '../utils/ErrorHandler.js';
import { SynapseAPI } from '../utils/SynapseAPI.js';
import { WebhookValidator } from '../utils/WebhookValidator.js';

export class WebhookHandler {
  constructor(env, ctx, logger) {
    this.env = env;
    this.ctx = ctx;
    this.logger = logger;
    this.synapseAPI = new SynapseAPI(env, logger);
    this.validator = new WebhookValidator(env, logger);
  }
  
  /**
   * Handle GitHub webhooks
   */
  async handleGitHub(request) {
    try {
      const signature = request.headers.get('X-Hub-Signature-256');
      const event = request.headers.get('X-GitHub-Event');
      const body = await request.text();
      
      this.logger.info('GitHub webhook received', {
        event,
        hasSignature: !!signature,
        bodyLength: body.length
      });
      
      // Validate webhook signature
      if (!this.validator.validateGitHubSignature(body, signature)) {
        return ErrorHandler.unauthorized('Invalid webhook signature');
      }
      
      const payload = JSON.parse(body);
      const tasks = [];
      
      // Process different GitHub events
      switch (event) {
        case 'push':
          tasks.push(...this.handleGitHubPush(payload));
          break;
          
        case 'pull_request':
          tasks.push(...this.handleGitHubPullRequest(payload));
          break;
          
        case 'issues':
          tasks.push(...this.handleGitHubIssues(payload));
          break;
          
        case 'release':
          tasks.push(...this.handleGitHubRelease(payload));
          break;
          
        default:
          this.logger.info('Unhandled GitHub event', { event });
          return new Response(JSON.stringify({
            success: true,
            message: `Event ${event} received but not processed`
          }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
          });
      }
      
      // Submit tasks
      const results = await this.submitTasks(tasks);
      
      return new Response(JSON.stringify({
        success: true,
        message: `GitHub ${event} webhook processed`,
        tasks_submitted: results.length,
        task_ids: results.map(r => r.task_id)
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error handling GitHub webhook', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to process GitHub webhook');
    }
  }
  
  /**
   * Handle generic webhooks
   */
  async handleGeneric(request) {
    try {
      const signature = request.headers.get('X-Webhook-Signature');
      const body = await request.text();
      
      this.logger.info('Generic webhook received', {
        hasSignature: !!signature,
        bodyLength: body.length
      });
      
      // Validate webhook signature if provided
      if (signature && !this.validator.validateGenericSignature(body, signature)) {
        return ErrorHandler.unauthorized('Invalid webhook signature');
      }
      
      const payload = JSON.parse(body);
      
      // Extract task information from payload
      const tasks = this.extractTasksFromGenericPayload(payload);
      
      if (tasks.length === 0) {
        return new Response(JSON.stringify({
          success: true,
          message: 'Webhook received but no tasks generated'
        }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      
      // Submit tasks
      const results = await this.submitTasks(tasks);
      
      return new Response(JSON.stringify({
        success: true,
        message: 'Generic webhook processed',
        tasks_submitted: results.length,
        task_ids: results.map(r => r.task_id)
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error handling generic webhook', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to process generic webhook');
    }
  }
  
  /**
   * Handle feed update webhooks
   */
  async handleFeedUpdate(request) {
    try {
      const body = await request.json();
      
      this.logger.info('Feed update webhook received', {
        feed_url: body.feed_url,
        items_count: body.items?.length || 0
      });
      
      // Validate required fields
      if (!body.feed_url) {
        return ErrorHandler.badRequest('Missing required field: feed_url');
      }
      
      const tasks = [];
      
      // Create task to process the updated feed
      tasks.push({
        task_type: 'process_feed',
        payload: {
          feed_url: body.feed_url,
          force_refresh: true,
          items: body.items || [],
          updated_at: body.updated_at || new Date().toISOString()
        },
        priority: 2, // High priority for feed updates
        source: 'feed_webhook'
      });
      
      // If specific items are provided, create scraping tasks
      if (body.items && Array.isArray(body.items)) {
        for (const item of body.items.slice(0, 10)) { // Limit to 10 items
          if (item.url) {
            tasks.push({
              task_type: 'scrape_url',
              payload: {
                url: item.url,
                title: item.title,
                feed_url: body.feed_url,
                published_at: item.published_at
              },
              priority: 3, // Normal priority for individual items
              source: 'feed_webhook'
            });
          }
        }
      }
      
      // Submit tasks
      const results = await this.submitTasks(tasks);
      
      return new Response(JSON.stringify({
        success: true,
        message: 'Feed update webhook processed',
        tasks_submitted: results.length,
        task_ids: results.map(r => r.task_id)
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
      
    } catch (error) {
      this.logger.error('Error handling feed update webhook', {
        error: error.message,
        stack: error.stack
      });
      return ErrorHandler.internalError('Failed to process feed update webhook');
    }
  }
  
  /**
   * Handle GitHub push events
   */
  handleGitHubPush(payload) {
    const tasks = [];
    
    // If README or documentation files were updated, trigger content analysis
    const docFiles = payload.commits?.flatMap(commit => 
      commit.modified?.filter(file => 
        file.toLowerCase().includes('readme') || 
        file.toLowerCase().includes('doc') ||
        file.endsWith('.md')
      ) || []
    ) || [];
    
    if (docFiles.length > 0) {
      tasks.push({
        task_type: 'analyze_content',
        payload: {
          repository: payload.repository?.full_name,
          files: docFiles,
          commit_sha: payload.head_commit?.id,
          commit_message: payload.head_commit?.message
        },
        priority: 4, // Low priority
        source: 'github_push'
      });
    }
    
    return tasks;
  }
  
  /**
   * Handle GitHub pull request events
   */
  handleGitHubPullRequest(payload) {
    const tasks = [];
    
    if (payload.action === 'opened' || payload.action === 'synchronize') {
      tasks.push({
        task_type: 'analyze_content',
        payload: {
          repository: payload.repository?.full_name,
          pull_request_number: payload.pull_request?.number,
          title: payload.pull_request?.title,
          body: payload.pull_request?.body,
          action: payload.action
        },
        priority: 3, // Normal priority
        source: 'github_pr'
      });
    }
    
    return tasks;
  }
  
  /**
   * Handle GitHub issues events
   */
  handleGitHubIssues(payload) {
    const tasks = [];
    
    if (payload.action === 'opened') {
      tasks.push({
        task_type: 'analyze_content',
        payload: {
          repository: payload.repository?.full_name,
          issue_number: payload.issue?.number,
          title: payload.issue?.title,
          body: payload.issue?.body,
          labels: payload.issue?.labels?.map(l => l.name) || []
        },
        priority: 4, // Low priority
        source: 'github_issues'
      });
    }
    
    return tasks;
  }
  
  /**
   * Handle GitHub release events
   */
  handleGitHubRelease(payload) {
    const tasks = [];
    
    if (payload.action === 'published') {
      tasks.push({
        task_type: 'analyze_content',
        payload: {
          repository: payload.repository?.full_name,
          release_tag: payload.release?.tag_name,
          release_name: payload.release?.name,
          release_body: payload.release?.body,
          prerelease: payload.release?.prerelease
        },
        priority: 2, // High priority for releases
        source: 'github_release'
      });
    }
    
    return tasks;
  }
  
  /**
   * Extract tasks from generic webhook payload
   */
  extractTasksFromGenericPayload(payload) {
    const tasks = [];
    
    // Look for common patterns in generic webhooks
    if (payload.url) {
      tasks.push({
        task_type: 'scrape_url',
        payload: {
          url: payload.url,
          title: payload.title,
          metadata: payload.metadata || {}
        },
        priority: payload.priority || 3,
        source: 'generic_webhook'
      });
    }
    
    if (payload.urls && Array.isArray(payload.urls)) {
      for (const url of payload.urls.slice(0, 20)) { // Limit to 20 URLs
        tasks.push({
          task_type: 'scrape_url',
          payload: {
            url: typeof url === 'string' ? url : url.url,
            title: typeof url === 'object' ? url.title : undefined,
            metadata: typeof url === 'object' ? url.metadata : {}
          },
          priority: payload.priority || 3,
          source: 'generic_webhook'
        });
      }
    }
    
    if (payload.task_type && payload.task_payload) {
      tasks.push({
        task_type: payload.task_type,
        payload: payload.task_payload,
        priority: payload.priority || 3,
        source: 'generic_webhook'
      });
    }
    
    return tasks;
  }
  
  /**
   * Submit multiple tasks
   */
  async submitTasks(tasks) {
    const results = [];
    
    for (const task of tasks) {
      try {
        const result = await this.synapseAPI.submitTask(task);
        if (result.success) {
          results.push(result);
          
          // Store in KV for tracking
          await this.env.TASK_STORE.put(
            `task:${result.task_id}`,
            JSON.stringify({
              ...task,
              task_id: result.task_id,
              status: 'submitted',
              created_at: new Date().toISOString()
            }),
            { expirationTtl: 86400 * 7 } // 7 days
          );
        }
      } catch (error) {
        this.logger.error('Failed to submit task', {
          task_type: task.task_type,
          error: error.message
        });
      }
    }
    
    return results;
  }
}