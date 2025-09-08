import React, { useState } from 'react';
import {
  Paper,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Alert,
  CircularProgress,
  Link
} from '@mui/material';
import { Key, Settings } from '@mui/icons-material';
import { configureApi } from '../services/api';

interface ApiConfigProps {
  onConfigured: () => void;
  onError: (error: string) => void;
}

const modelOptions = {
  'Gemma 9B (Efficient)': 'gemma2-9b-it'
};

const ApiConfig: React.FC<ApiConfigProps> = ({ onConfigured, onError }) => {
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('Gemma 9B (Efficient)');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!apiKey.trim()) {
      onError('Please enter your Groq API key');
      return;
    }

    setLoading(true);

    try {
      const modelName = modelOptions[selectedModel as keyof typeof modelOptions];
      const response = await configureApi(apiKey, modelName);
      
      if (response.error) {
        onError(response.error);
      } else {
        onConfigured();
      }
    } catch (error: any) {
      onError(error.response?.data?.error || 'Failed to configure API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 600, mx: 'auto' }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Settings sx={{ mr: 2, fontSize: 32, color: 'primary.main' }} />
          <Typography variant="h4" component="h1">
            API Configuration
          </Typography>
        </Box>

        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            <strong>To get started:</strong>
            <br />
            1. Go to{' '}
            <Link 
              href="https://console.groq.com/keys" 
              target="_blank" 
              rel="noopener noreferrer"
            >
              https://console.groq.com/keys
            </Link>
            <br />
            2. Create a free account
            <br />
            3. Generate an API key
            <br />
            4. Enter it below
          </Typography>
        </Alert>

        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Groq API Key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your Groq API key"
            InputProps={{
              startAdornment: <Key sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
            margin="normal"
            required
            helperText="Your API key is used locally and never stored permanently"
          />

          <FormControl fullWidth margin="normal">
            <InputLabel>Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              label="Model"
            >
              {Object.keys(modelOptions).map((modelName) => (
                <MenuItem key={modelName} value={modelName}>
                  {modelName}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            type="submit"
            fullWidth
            variant="contained"
            size="large"
            disabled={loading || !apiKey.trim()}
            sx={{ mt: 3, py: 1.5 }}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              'Configure System'
            )}
          </Button>
        </form>

        <Box sx={{ mt: 3 }}>
          <Typography variant="body2" color="text.secondary">
            <strong>Model Information:</strong>
          </Typography>
          <Typography variant="caption" color="text.secondary" component="div">
            • <strong>Gemma 9B:</strong> Efficient and lightweight model for research paper analysis
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default ApiConfig;