/**
 * Vercel Edge Function - Task Fallback Handler
 * 
 * This function serves as a fallback when Cloudflare Workers are unavailable.
 * It provides basic task management functionality with reduced features.
 */

import { NextRequest, NextResponse } from 'next/server';

export const config = {
  runtime: 'edge',
};

// In-memory storage for fallback (in production, use a database)
const fallbackTasks = new Map();
let taskCounter = 0;

export default async function handler(request) {
  const { pathname, searchParams } = new URL(request.url);
  const method = request.method;
  const pathSegments = pathname.split('/').filter(Boolean);
  
  // Remove 'api' and 'tasks' from path segments
  const taskPath = pathSegments.slice(2);
  
  console.log('Vercel fallback handler:', { method, pathname, taskPath });
  
  try {
    // Handle CORS preflight
    if (method === 'OPTIONS') {
      return new NextResponse(null, {
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-API-Key',
          'Access-Control-Max-Age': '86400'
        }
      });
    }
    
    // Route handling
    if (method === 'POST' && taskPath.length === 0) {
      return await handleSubmitTask(request);
    }
    
    if (method === 'GET' && taskPath.length === 1) {
      return await handleGetTaskStatus(request, taskPath[0]);
    }
    
    if (method === 'DELETE' && taskPath.length === 1) {
      return await handleCancelTask(request, taskPath[0]);
    }
    
    if (method === 'GET' && taskPath.length === 0) {
      return await handleListTasks(request);
    }
    
    return new NextResponse(JSON.stringify({
      success: false,
      error: 'Endpoint not found',
      message: 'This is a fallback service with limited functionality'
    }), {
      status: 404,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
    
  } catch (error) {
    console.error('Error in Vercel fallback handler:', error);
    
    return new NextResponse(JSON.stringify({
      success: false,
      error: 'Internal server error',
      message: 'Fallback service error'
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
  }
}

/**
 * Handle task submission
 */
async function handleSubmitTask(request) {
  try {
    const body = await request.json();
    
    // Validate required fields
    if (!body.task_type || !body.payload) {
      return new NextResponse(JSON.stringify({
        success: false,
        error: 'Missing required fields: task_type, payload'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Generate task ID
    const taskId = `fallback-${Date.now()}-${++taskCounter}`;
    
    // Create task
    const task = {
      id: taskId,
      task_type: body.task_type,
      payload: body.payload,
      priority: body.priority || 3,
      max_retries: body.max_retries || 3,
      status: 'queued',
      created_at: new Date().toISOString(),
      source: 'vercel_fallback',
      fallback: true
    };
    
    // Store task (in production, use a database)
    fallbackTasks.set(taskId, task);
    
    console.log('Task submitted to fallback:', { taskId, task_type: body.task_type });
    
    return new NextResponse(JSON.stringify({
      success: true,
      task_id: taskId,
      status: 'queued',
      message: 'Task submitted to fallback service',
      fallback: true
    }), {
      status: 201,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
    
  } catch (error) {
    console.error('Error submitting task to fallback:', error);
    
    return new NextResponse(JSON.stringify({
      success: false,
      error: 'Failed to submit task to fallback service'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle get task status
 */
async function handleGetTaskStatus(request, taskId) {
  try {
    const task = fallbackTasks.get(taskId);
    
    if (!task) {
      return new NextResponse(JSON.stringify({
        success: false,
        error: 'Task not found',
        message: 'Task not found in fallback service'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    return new NextResponse(JSON.stringify({
      success: true,
      data: {
        ...task,
        fallback_service: true,
        note: 'This task is stored in fallback service with limited functionality'
      }
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
    
  } catch (error) {
    console.error('Error getting task status from fallback:', error);
    
    return new NextResponse(JSON.stringify({
      success: false,
      error: 'Failed to get task status from fallback service'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle cancel task
 */
async function handleCancelTask(request, taskId) {
  try {
    const task = fallbackTasks.get(taskId);
    
    if (!task) {
      return new NextResponse(JSON.stringify({
        success: false,
        error: 'Task not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }
    
    // Update task status
    task.status = 'cancelled';
    task.cancelled_at = new Date().toISOString();
    fallbackTasks.set(taskId, task);
    
    return new NextResponse(JSON.stringify({
      success: true,
      message: 'Task cancelled in fallback service'
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
    
  } catch (error) {
    console.error('Error cancelling task in fallback:', error);
    
    return new NextResponse(JSON.stringify({
      success: false,
      error: 'Failed to cancel task in fallback service'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}

/**
 * Handle list tasks
 */
async function handleListTasks(request) {
  try {
    const url = new URL(request.url);
    const limit = parseInt(url.searchParams.get('limit')) || 50;
    const offset = parseInt(url.searchParams.get('offset')) || 0;
    const status = url.searchParams.get('status');
    const task_type = url.searchParams.get('task_type');
    
    let tasks = Array.from(fallbackTasks.values());
    
    // Apply filters
    if (status) {
      tasks = tasks.filter(task => task.status === status);
    }
    
    if (task_type) {
      tasks = tasks.filter(task => task.task_type === task_type);
    }
    
    // Sort by created_at (newest first)
    tasks.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
    // Apply pagination
    const total = tasks.length;
    const paginatedTasks = tasks.slice(offset, offset + limit);
    
    return new NextResponse(JSON.stringify({
      success: true,
      data: paginatedTasks,
      pagination: {
        limit,
        offset,
        total
      },
      fallback_service: true,
      note: 'Tasks listed from fallback service with limited functionality'
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
    
  } catch (error) {
    console.error('Error listing tasks from fallback:', error);
    
    return new NextResponse(JSON.stringify({
      success: false,
      error: 'Failed to list tasks from fallback service'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}