/**
 * Health check endpoint for Spinal Cord
 */
export default function handler(req, res) {
  res.status(200).json({
    status: 'healthy',
    component: 'spinal_cord',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    environment: process.env.ENVIRONMENT || 'production'
  });
}