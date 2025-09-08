import React, { useState, useCallback } from 'react';
import {
  Paper,
  Typography,
  Box,
  Button,
  TextField,
  Grid,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Divider,
  Chip
} from '@mui/material';
import {
  Upload as UploadIcon,
  Description,
  CloudDownload,
  HourglassEmpty,
  Delete,
  Science
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { SystemStatus } from '../types';
import { uploadFile, downloadArxivPaper, processPapers, clearPapers } from '../services/api';

interface PaperManagementProps {
  systemStatus: SystemStatus;
  onStatusUpdate: () => void;
  onMessage: (message: string, severity?: 'info' | 'success' | 'warning' | 'error') => void;
}

const PaperManagement: React.FC<PaperManagementProps> = ({
  systemStatus,
  onStatusUpdate,
  onMessage
}) => {
  const [arxivId, setArxivId] = useState('');
  const [loading, setLoading] = useState({ upload: false, arxiv: false, process: false, clear: false });

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setLoading(prev => ({ ...prev, upload: true }));
    
    try {
      for (const file of acceptedFiles) {
        if (file.type === 'application/pdf') {
          const response = await uploadFile(file);
          if (response.error) {
            onMessage(response.error, 'error');
          } else {
            onMessage(`${file.name} uploaded successfully`, 'success');
          }
        } else {
          onMessage(`${file.name} is not a PDF file`, 'warning');
        }
      }
      onStatusUpdate();
    } catch (error: any) {
      onMessage(error.response?.data?.error || 'Upload failed', 'error');
    } finally {
      setLoading(prev => ({ ...prev, upload: false }));
    }
  }, [onMessage, onStatusUpdate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: true
  });

  const handleArxivDownload = async () => {
    if (!arxivId.trim()) {
      onMessage('Please enter an arXiv ID', 'warning');
      return;
    }

    setLoading(prev => ({ ...prev, arxiv: true }));

    try {
      const response = await downloadArxivPaper(arxivId);
      if (response.error) {
        onMessage(response.error, 'error');
      } else {
        onMessage('Paper downloaded successfully', 'success');
        setArxivId('');
        onStatusUpdate();
      }
    } catch (error: any) {
      onMessage(error.response?.data?.error || 'Download failed', 'error');
    } finally {
      setLoading(prev => ({ ...prev, arxiv: false }));
    }
  };

  const handleProcessPapers = async () => {
    if (systemStatus.papers.length === 0) {
      onMessage('No papers to process', 'warning');
      return;
    }

    setLoading(prev => ({ ...prev, process: true }));

    try {
      const response = await processPapers();
      if (response.error) {
        onMessage(response.error, 'error');
      } else {
        onMessage(response.success || 'Papers processed successfully', 'success');
        onStatusUpdate();
      }
    } catch (error: any) {
      onMessage(error.response?.data?.error || 'Processing failed', 'error');
    } finally {
      setLoading(prev => ({ ...prev, process: false }));
    }
  };

  const handleClearPapers = async () => {
    if (systemStatus.papers.length === 0) {
      onMessage('No papers to clear', 'info');
      return;
    }

    setLoading(prev => ({ ...prev, clear: true }));

    try {
      const response = await clearPapers();
      if (response.error) {
        onMessage(response.error, 'error');
      } else {
        onMessage('All papers cleared', 'success');
        onStatusUpdate();
      }
    } catch (error: any) {
      onMessage(error.response?.data?.error || 'Clear failed', 'error');
    } finally {
      setLoading(prev => ({ ...prev, clear: false }));
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Science sx={{ mr: 2, fontSize: 32, color: 'primary.main' }} />
        <Typography variant="h4" component="h1">
          Paper Management
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Upload Section */}
        <Grid xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload PDF Files
              </Typography>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 3,
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: isDragActive ? 'action.hover' : 'background.default',
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  '&:hover': {
                    backgroundColor: 'action.hover',
                    borderColor: 'primary.main'
                  }
                }}
              >
                <input {...getInputProps()} />
                <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body1" gutterBottom>
                  {isDragActive
                    ? 'Drop the PDF files here...'
                    : 'Drag & drop PDF files here, or click to select'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supports multiple PDF files
                </Typography>
                {loading.upload && (
                  <Box sx={{ mt: 2 }}>
                    <CircularProgress size={24} />
                  </Box>
                )}
              </Paper>
            </CardContent>
          </Card>
        </Grid>

        {/* ArXiv Download Section */}
        <Grid xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Download from arXiv
              </Typography>
              <TextField
                fullWidth
                label="arXiv ID"
                value={arxivId}
                onChange={(e) => setArxivId(e.target.value)}
                placeholder="e.g., 2301.00001"
                margin="normal"
                helperText="Enter the arXiv paper ID"
              />
              <Button
                fullWidth
                variant="outlined"
                onClick={handleArxivDownload}
                disabled={loading.arxiv || !arxivId.trim()}
                sx={{ mt: 2 }}
                startIcon={loading.arxiv ? <CircularProgress size={20} /> : <CloudDownload />}
              >
                {loading.arxiv ? 'Downloading...' : 'Download Paper'}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Papers Section */}
        <Grid xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Typography variant="h6">
                  Loaded Papers ({systemStatus.papers.length})
                </Typography>
                <Box>
                  <Chip 
                    label={systemStatus.ready ? "Ready" : "Not Processed"} 
                    color={systemStatus.ready ? "success" : "warning"} 
                    sx={{ mr: 1 }}
                  />
                </Box>
              </Box>
              
              {systemStatus.papers.length === 0 ? (
                <Typography color="text.secondary" sx={{ py: 2 }}>
                  No papers loaded. Upload PDF files or download from arXiv to get started.
                </Typography>
              ) : (
                <List>
                  {systemStatus.papers.map((paper, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <Description color="primary" />
                      </ListItemIcon>
                      <ListItemText
                        primary={paper}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              )}

              {systemStatus.papers.length > 0 && (
                <>
                  <Divider sx={{ my: 2 }} />
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button
                      variant="contained"
                      onClick={handleProcessPapers}
                      disabled={loading.process || systemStatus.ready}
                      startIcon={loading.process ? <CircularProgress size={20} /> : <HourglassEmpty />}
                      sx={{ flex: 1 }}
                    >
                      {loading.process ? 'Processing...' : systemStatus.ready ? 'Papers Processed' : 'Process Papers'}
                    </Button>
                    <Button
                      variant="outlined"
                      color="error"
                      onClick={handleClearPapers}
                      disabled={loading.clear}
                      startIcon={loading.clear ? <CircularProgress size={20} /> : <Delete />}
                    >
                      {loading.clear ? 'Clearing...' : 'Clear All'}
                    </Button>
                  </Box>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PaperManagement;