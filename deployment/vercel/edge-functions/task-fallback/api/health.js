/**
 * Vercel Edge Function - Health Check
 * 
 * Provides health check endpoint for the fallback service
 */

import { NextResponse } from 'next/server';

export const config = {
  runtime: 'edge',
};

export default async function handler(request) {
  const startTime = Date.now();
  
  try {
    // Basic health check
    const health = {
      status: 'healthy',
      service: 'vercel-edge-fallback',
      timestamp: new Date().toISOString(),
      uptime: process.uptime ? process.uptime() : 'unknown',
      version: '1.0.0',
      environment: process.env.VERCEL_ENV || 'unknown',
      region: process.env.VERCEL_REGION || 'unknown',
      response_time_ms: Date.now() - startTime
    };
    
    return new NextResponse(JSON.stringify(health), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
    
  } catch (error) {
    console.error('Health check error:', error);
    
    return new NextResponse(JSON.stringify({
      status: 'unhealthy',
      service: 'vercel-edge-fallback',
      error: error.message,
      timestamp: new Date().toISOString()
    }), {
      status: 503,
      headers: {
        'Content-Type': 'application/json',
        'X-Served-By': 'vercel-edge-fallback'
      }
    });
  }
}