import axios from 'axios';
import { SystemStatus, ApiResponse, QuestionResponse } from '../types';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const configureApi = async (groqApiKey: string, modelName: string = 'llama3-8b-8192'): Promise<ApiResponse> => {
  const response = await api.post('/config', {
    groq_api_key: groqApiKey,
    model_name: modelName,
  });
  return response.data;
};

export const getSystemStatus = async (): Promise<SystemStatus> => {
  const response = await api.get('/status');
  return response.data;
};

export const uploadFile = async (file: File): Promise<ApiResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const downloadArxivPaper = async (arxivId: string): Promise<ApiResponse> => {
  const response = await api.post('/download-arxiv', {
    arxiv_id: arxivId,
  });
  return response.data;
};

export const processPapers = async (): Promise<ApiResponse> => {
  const response = await api.post('/process-papers');
  return response.data;
};

export const askQuestion = async (question: string, useChatEngine: boolean = true): Promise<QuestionResponse> => {
  const response = await api.post('/ask', {
    question,
    use_chat_engine: useChatEngine,
  });
  return response.data;
};

export const clearPapers = async (): Promise<ApiResponse> => {
  const response = await api.post('/clear-papers');
  return response.data;
};

export const clearChatHistory = async (): Promise<ApiResponse> => {
  const response = await api.post('/clear-chat');
  return response.data;
};