export interface SystemStatus {
  configured: boolean;
  ready: boolean;
  papers: string[];
  chat_history: ChatMessage[];
}

export interface ChatMessage {
  timestamp: string;
  question: string;
  answer: string;
  type: 'chat' | 'query';
}

export interface ApiResponse {
  success?: string;
  error?: string;
}

export interface QuestionResponse extends ApiResponse {
  answer?: string;
  sources?: Source[];
  timestamp?: string;
}

export interface Source {
  text: string;
  score: string | number;
}