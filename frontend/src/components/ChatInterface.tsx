import React, { useState, useRef, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  Alert
} from '@mui/material';
import {
  Send,
  Chat,
  ExpandMore,
  Clear,
  QuestionAnswer,
  Science as ScienceIcon,
  Psychology,
  Analytics,
  Task,
  Warning,
  Summarize
} from '@mui/icons-material';
import { SystemStatus, ChatMessage } from '../types';
import { askQuestion, clearChatHistory } from '../services/api';

interface ChatInterfaceProps {
  systemStatus: SystemStatus;
  onStatusUpdate: () => void;
  onMessage: (message: string, severity?: 'info' | 'success' | 'warning' | 'error') => void;
}

const quickQuestions = [
  {
    icon: <QuestionAnswer />,
    title: 'Main Research Question',
    question: 'What is the main research question addressed in this paper?'
  },
  {
    icon: <ScienceIcon />,
    title: 'Methodology',
    question: 'What methodology was used in this study?'
  },
  {
    icon: <Analytics />,
    title: 'Key Findings',
    question: 'What are the key findings of this research?'
  },
  {
    icon: <Task />,
    title: 'Conclusions',
    question: 'What are the main conclusions of this research?'
  },
  {
    icon: <Warning />,
    title: 'Limitations',
    question: 'What are the limitations of this study?'
  },
  {
    icon: <Summarize />,
    title: 'Summary',
    question: 'Please provide a summary of this paper.'
  }
];

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  systemStatus,
  onStatusUpdate,
  onMessage
}) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [clearingChat, setClearingChat] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [systemStatus.chat_history]);

  const handleSendQuestion = async (questionText: string) => {
    if (!questionText.trim() || loading) return;

    setLoading(true);
    try {
      const response = await askQuestion(questionText, true);
      if (response.error) {
        onMessage(response.error, 'error');
      } else {
        onStatusUpdate();
        setQuestion('');
      }
    } catch (error: any) {
      onMessage(error.response?.data?.error || 'Failed to send question', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendQuestion(question);
  };

  const handleQuickQuestion = (quickQuestion: string) => {
    handleSendQuestion(quickQuestion);
  };

  const handleClearChat = async () => {
    setClearingChat(true);
    try {
      const response = await clearChatHistory();
      if (response.error) {
        onMessage(response.error, 'error');
      } else {
        onMessage('Chat history cleared', 'success');
        onStatusUpdate();
      }
    } catch (error: any) {
      onMessage(error.response?.data?.error || 'Failed to clear chat', 'error');
    } finally {
      setClearingChat(false);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(`2000-01-01T${timestamp}`).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  if (!systemStatus.ready) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Psychology sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          System Not Ready
        </Typography>
        <Typography color="text.secondary">
          Please process your papers first before starting a chat.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Chat sx={{ mr: 2, fontSize: 32, color: 'primary.main' }} />
          <Typography variant="h4" component="h1">
            Chat with Your Papers
          </Typography>
        </Box>
        <IconButton 
          onClick={handleClearChat}
          disabled={clearingChat || systemStatus.chat_history.length === 0}
          color="error"
          title="Clear Chat History"
        >
          {clearingChat ? <CircularProgress size={24} /> : <Clear />}
        </IconButton>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          📚 Chatting with {systemStatus.papers.length} paper(s): {systemStatus.papers.join(', ')}
        </Typography>
      </Alert>

      {/* Quick Questions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🚀 Quick Questions
          </Typography>
          <Grid container spacing={2}>
            {quickQuestions.map((item, index) => (
              <Grid xs={12} sm={6} md={4} key={index}>
                <Button
                  fullWidth
                  variant="outlined"
                  startIcon={item.icon}
                  onClick={() => handleQuickQuestion(item.question)}
                  disabled={loading}
                  sx={{ 
                    justifyContent: 'flex-start', 
                    textAlign: 'left',
                    height: 56
                  }}
                >
                  <Typography variant="body2">
                    {item.title}
                  </Typography>
                </Button>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Chat History */}
      <Card sx={{ mb: 3, height: 400, display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ pb: 1 }}>
          <Typography variant="h6" gutterBottom>
            💬 Conversation
          </Typography>
        </CardContent>
        <Box 
          ref={chatContainerRef}
          sx={{ 
            flex: 1, 
            overflowY: 'auto', 
            px: 2, 
            pb: 2,
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: 'rgba(0,0,0,.2)',
              borderRadius: '4px',
            },
          }}
        >
          {systemStatus.chat_history.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4, color: 'text.secondary' }}>
              <Psychology sx={{ fontSize: 48, mb: 2 }} />
              <Typography>
                No messages yet. Ask a question or try one of the quick questions above!
              </Typography>
            </Box>
          ) : (
            systemStatus.chat_history.map((message: ChatMessage, index: number) => (
              <Box key={index} sx={{ mb: 3 }}>
                {/* User Question */}
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                  <Paper 
                    sx={{ 
                      p: 2, 
                      maxWidth: '80%', 
                      backgroundColor: 'primary.main',
                      color: 'primary.contrastText'
                    }}
                  >
                    <Typography variant="body1">
                      {message.question}
                    </Typography>
                    <Typography variant="caption" sx={{ opacity: 0.8, mt: 1, display: 'block' }}>
                      {formatTimestamp(message.timestamp)}
                    </Typography>
                  </Paper>
                </Box>

                {/* Bot Answer */}
                <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                  <Paper 
                    sx={{ 
                      p: 2, 
                      maxWidth: '80%', 
                      backgroundColor: 'grey.100'
                    }}
                  >
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                      {message.answer}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, gap: 1 }}>
                      <Chip 
                        size="small" 
                        label={message.type === 'chat' ? 'Chat Mode' : 'Q&A Mode'} 
                        color={message.type === 'chat' ? 'primary' : 'secondary'}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {formatTimestamp(message.timestamp)}
                      </Typography>
                    </Box>
                  </Paper>
                </Box>

                {index < systemStatus.chat_history.length - 1 && <Divider sx={{ my: 2 }} />}
              </Box>
            ))
          )}
        </Box>
      </Card>

      {/* Question Input */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            💭 Ask Your Question
          </Typography>
          <form onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
              <TextField
                fullWidth
                multiline
                maxRows={4}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask anything about your papers..."
                disabled={loading}
                variant="outlined"
              />
              <Button
                type="submit"
                variant="contained"
                disabled={loading || !question.trim()}
                sx={{ minWidth: 120, height: 56 }}
                startIcon={loading ? <CircularProgress size={20} /> : <Send />}
              >
                {loading ? 'Thinking...' : 'Send'}
              </Button>
            </Box>
          </form>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            💬 Conversational Mode: ON - Your questions will build on previous conversation context
          </Typography>
        </CardContent>
      </Card>

    </Box>
  );
};

export default ChatInterface;