import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Divider,
  Alert,
  Snackbar
} from '@mui/material';
import {
  School,
  Settings,
  Chat,
  Upload,
  Menu as MenuIcon,
  Science
} from '@mui/icons-material';

import ApiConfig from './components/ApiConfig';
import ChatInterface from './components/ChatInterface';
import PaperManagement from './components/PaperManagement';
import { SystemStatus } from './types';
import { getSystemStatus } from './services/api';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

const drawerWidth = 240;

function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [currentView, setCurrentView] = useState<'config' | 'papers' | 'chat'>('config');
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    configured: false,
    ready: false,
    papers: [],
    chat_history: []
  });
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' as 'info' | 'success' | 'warning' | 'error' });

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const showSnackbar = (message: string, severity: 'info' | 'success' | 'warning' | 'error' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const closeSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const updateSystemStatus = async () => {
    try {
      const status = await getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to update system status:', error);
    }
  };

  useEffect(() => {
    updateSystemStatus();
    const interval = setInterval(updateSystemStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Only auto-navigate from config to papers if system gets configured
    if (systemStatus.configured && !systemStatus.ready && currentView === 'config') {
      setCurrentView('papers');
    }
    // Auto-navigate from papers to chat only if we're still on papers and system becomes ready
    // but allow manual navigation back to papers even when ready
  }, [systemStatus, currentView]);

  const drawer = (
    <div>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Science color="primary" />
          <Typography variant="h6" noWrap component="div" color="primary">
            Research Assistant
          </Typography>
        </Box>
      </Toolbar>
      <Divider />
      <List>
        <ListItem 
          button
          onClick={() => setCurrentView('config')}
          selected={currentView === 'config'}
        >
          <ListItemIcon>
            <Settings />
          </ListItemIcon>
          <ListItemText primary="Configuration" />
        </ListItem>
        <ListItem 
          button
          onClick={() => setCurrentView('papers')}
          selected={currentView === 'papers'}
        >
          <ListItemIcon>
            <Upload />
          </ListItemIcon>
          <ListItemText primary="Papers" />
        </ListItem>
        <ListItem 
          button
          onClick={() => setCurrentView('chat')}
          selected={currentView === 'chat'}
          disabled={!systemStatus.ready}
        >
          <ListItemIcon>
            <Chat />
          </ListItemIcon>
          <ListItemText primary="Chat" />
        </ListItem>
      </List>
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" display="block" gutterBottom>
          Status:
        </Typography>
        <Typography 
          variant="body2" 
          color={systemStatus.configured ? 'success.main' : 'text.secondary'}
        >
          {systemStatus.configured ? '✅ Configured' : '⚠️ Not Configured'}
        </Typography>
        <Typography 
          variant="body2" 
          color={systemStatus.ready ? 'success.main' : 'text.secondary'}
        >
          {systemStatus.ready ? '✅ Ready' : '⚠️ Process Papers'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Papers: {systemStatus.papers.length}
        </Typography>
      </Box>
    </div>
  );

  const renderCurrentView = () => {
    switch (currentView) {
      case 'config':
        return (
          <ApiConfig 
            onConfigured={() => {
              updateSystemStatus();
              showSnackbar('System configured successfully!', 'success');
            }}
            onError={(error) => showSnackbar(error, 'error')}
          />
        );
      case 'papers':
        return (
          <PaperManagement 
            systemStatus={systemStatus}
            onStatusUpdate={updateSystemStatus}
            onMessage={showSnackbar}
          />
        );
      case 'chat':
        return (
          <ChatInterface 
            systemStatus={systemStatus}
            onStatusUpdate={updateSystemStatus}
            onMessage={showSnackbar}
          />
        );
      default:
        return null;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        <AppBar
          position="fixed"
          sx={{
            width: { sm: `calc(100% - ${drawerWidth}px)` },
            ml: { sm: `${drawerWidth}px` },
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { sm: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <School sx={{ mr: 2 }} />
            <Typography variant="h6" noWrap component="div">
              Academic Paper Q&A Bot (Groq Powered)
            </Typography>
          </Toolbar>
        </AppBar>
        <Box
          component="nav"
          sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        >
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{
              keepMounted: true,
            }}
            sx={{
              display: { xs: 'block', sm: 'none' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
          >
            {drawer}
          </Drawer>
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: 'none', sm: 'block' },
              '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
            }}
            open
          >
            {drawer}
          </Drawer>
        </Box>
        <Box
          component="main"
          sx={{ 
            flexGrow: 1, 
            p: 3, 
            width: { sm: `calc(100% - ${drawerWidth}px)` },
            mt: 8
          }}
        >
          <Container maxWidth="lg">
            {renderCurrentView()}
          </Container>
        </Box>
        
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={closeSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={closeSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
