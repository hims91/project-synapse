/**
 * Store tasks in fallback storage (R2)
 */
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { tasks, timestamp, reason } = req.body;
    
    if (!tasks || !Array.isArray(tasks)) {
      return res.status(400).json({ error: 'Tasks array is required' });
    }

    // In a real implementation, this would store to Cloudflare R2
    console.log(`Storing ${tasks.length} tasks to fallback storage`);
    console.log(`Reason: ${reason}, Timestamp: ${timestamp}`);
    
    // Simulate successful storage
    res.status(200).json({
      status: 'success',
      stored_tasks: tasks.length,
      storage_id: `fallback_${Date.now()}`,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Error storing fallback tasks:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}