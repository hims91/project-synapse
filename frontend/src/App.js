import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  Box,
  Chip,
  CircularProgress
} from '@mui/material';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching system status:', error);
      setSystemStatus({ status: 'error', message: 'Unable to connect to hub' });
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading Project Synapse...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h2" component="h1" gutterBottom>
          🧠 Project Synapse
        </Typography>
        <Typography variant="h5" color="text.secondary">
          AI-Powered Content Intelligence Platform
        </Typography>
        <Chip 
          label={systemStatus?.status || 'Unknown'} 
          color={getStatusColor(systemStatus?.status)}
          sx={{ mt: 2 }}
        />
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Overall system health and component status
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Typography variant="body1">
                  Status: <strong>{systemStatus?.status || 'Unknown'}</strong>
                </Typography>
                <Typography variant="body1">
                  Version: <strong>{systemStatus?.version || 'N/A'}</strong>
                </Typography>
                <Typography variant="body1">
                  Last Updated: <strong>{new Date().toLocaleTimeString()}</strong>
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Components
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Status of distributed system components
              </Typography>
              <Box sx={{ mt: 2 }}>
                {systemStatus?.components ? (
                  Object.entries(systemStatus.components).map(([component, status]) => (
                    <Box key={component} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">{component}:</Typography>
                      <Chip 
                        label={status} 
                        size="small" 
                        color={getStatusColor(status)}
                      />
                    </Box>
                  ))
                ) : (
                  <Typography variant="body2">No component data available</Typography>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Architecture Overview
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Brain-inspired distributed system running across multiple cloud platforms
              </Typography>
              <Grid container spacing={2} sx={{ mt: 2 }}>
                <Grid item xs={12} sm={6} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="h6">🧠 Central Cortex</Typography>
                    <Typography variant="body2">Hub Server (Render)</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="h6">🌐 Dendrites</Typography>
                    <Typography variant="body2">Feed Pollers (Cloudflare)</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="h6">🕷️ Neurons</Typography>
                    <Typography variant="body2">Scrapers (Railway)</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="h6">🤖 Sensory Neurons</Typography>
                    <Typography variant="body2">Learning (GitHub)</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="h6">🦴 Spinal Cord</Typography>
                    <Typography variant="body2">Fallback (Vercel)</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                  <Box sx={{ textAlign: 'center', p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                    <Typography variant="h6">🎨 Frontend</Typography>
                    <Typography variant="body2">Dashboard (Netlify)</Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ textAlign: 'center', mt: 4, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
        <Typography variant="body2" color="text.secondary">
          Project Synapse - Distributed AI Intelligence Network
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Running on free-tier cloud platforms with zero infrastructure cost
        </Typography>
      </Box>
    </Container>
  );
}

export default App;