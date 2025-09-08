# Academic Paper Q&A Bot (React + Flask)

A modern web application that allows you to upload academic papers and chat with them using advanced AI. This project has been ported from Streamlit to a React frontend with a Flask backend, powered by Groq API for fast AI responses.

## ✨ Features

- **Modern UI**: Clean, responsive React interface with Material-UI components
- **Paper Upload**: Drag & drop PDF files or download directly from arXiv
- **AI-Powered Q&A**: Chat with your papers using state-of-the-art language models
- **Multiple Models**: Choose from Llama3, Mixtral, and Gemma models via Groq API
- **Conversational Chat**: Maintains context across questions for natural conversations
- **Real-time Status**: Live updates on system status and processing progress
- **Quick Questions**: Pre-built questions for common research inquiries


<img width="1623" height="915" alt="Screenshot 2025-09-08 211030" src="https://github.com/user-attachments/assets/15878c58-9e72-4e04-a22b-a9b460da4e97" />

<img width="1370" height="905" alt="Screenshot 2025-09-08 211052" src="https://github.com/user-attachments/assets/7ee1b016-b6e7-4301-bba6-0eaeed56a131" />

<img width="1491" height="759" alt="Screenshot 2025-09-08 211105" src="https://github.com/user-attachments/assets/ffb5701f-b342-4362-9440-1ea4555738fd" />

## 🏗️ Architecture

```
├── backend/          # Flask API server
│   ├── app.py       # Main Flask application
│   └── requirements.txt
├── frontend/         # React TypeScript application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API service layer
│   │   └── types.ts     # TypeScript interfaces
│   └── package.json
├── P2/              # Original Streamlit implementation
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Groq API key (free at https://console.groq.com/keys)

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask server**:
   ```bash
   python app.py
   ```

   The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm start
   ```

   The frontend will start on `http://localhost:3000`

## 🔧 Configuration

### Get Your Groq API Key

1. Visit https://console.groq.com/keys
2. Create a free account
3. Generate an API key
4. Enter it in the application's configuration page

### Available Models

- **Llama3 8B (Fast & Stable)**: Best for general use, fast responses
- **Llama3 70B (Most Capable)**: Highest quality responses, slower
- **Mixtral 8x7B (Balanced)**: Good balance of speed and quality
- **Gemma 7B (Efficient)**: Lightweight and efficient

## 📖 How to Use

### 1. Configure API
- Open the application at `http://localhost:3000`
- Enter your Groq API key
- Select your preferred model
- Click "Configure System"

### 2. Upload Papers
- Upload PDF files using drag & drop
- Or download papers directly from arXiv using paper IDs
- Click "Process Papers" to index the documents

### 3. Start Chatting
- Use quick question buttons for common queries
- Or type custom questions in the chat interface
- Enjoy conversational AI that remembers context

## 🔍 Quick Questions

The application provides pre-built questions for common research needs:

- **Main Research Question**: What is the main research question?
- **Methodology**: What methodology was used?
- **Key Findings**: What are the key findings?
- **Conclusions**: What are the main conclusions?
- **Limitations**: What are the limitations?
- **Summary**: Provide a comprehensive summary

## 🛠️ Technical Details

### Backend (Flask)
- **Framework**: Flask with Flask-CORS for cross-origin requests
- **AI Integration**: LlamaIndex for document processing and querying
- **Vector Storage**: In-memory vector storage for fast retrieval
- **File Handling**: Secure file uploads with Werkzeug
- **API**: RESTful API endpoints for all functionality

### Frontend (React)
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI) for modern components
- **State Management**: React hooks for state management
- **File Upload**: React Dropzone for drag & drop functionality
- **HTTP Client**: Axios for API communication

### Key Dependencies

#### Backend
- `Flask`: Web framework
- `llama-index`: Document indexing and querying
- `arxiv`: arXiv paper downloading
- `sentence-transformers`: Text embeddings
- `groq`: Groq API integration

#### Frontend
- `@mui/material`: UI components
- `axios`: HTTP client
- `react-dropzone`: File upload
- `@mui/icons-material`: Icons

## 🚦 API Endpoints

### Configuration
- `POST /api/config` - Configure Groq API and model
- `GET /api/status` - Get system status

### Paper Management
- `POST /api/upload` - Upload PDF file
- `POST /api/download-arxiv` - Download from arXiv
- `POST /api/process-papers` - Process uploaded papers
- `POST /api/clear-papers` - Clear all papers

### Chat
- `POST /api/ask` - Ask a question
- `POST /api/clear-chat` - Clear chat history

## 🔒 Security Features

- API key is only stored in memory (not persisted)
- Secure file upload with type validation
- CORS protection for cross-origin requests
- Input sanitization and validation

## 🚀 Production Deployment

### Backend
1. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Frontend
1. Build the React app:
   ```bash
   npm run build
   ```
2. Serve static files with nginx or similar web server

### Environment Variables
Set `GROQ_API_KEY` environment variable for production:
```bash
export GROQ_API_KEY=your_api_key_here
```

## 📊 Performance Optimizations

- **CPU-Optimized**: Configured for CPU-only inference
- **Batch Processing**: Documents processed in batches for efficiency
- **Memory Management**: Conservative token limits to prevent overflow
- **Caching**: Vector index caching for faster subsequent loads
- **Chunking**: Optimized text chunking for better retrieval

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**:
   - Check Python version (3.8+)
   - Ensure all dependencies are installed
   - Verify virtual environment is activated

2. **Frontend build fails**:
   - Check Node.js version (16+)
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`

3. **API connection issues**:
   - Ensure backend is running on port 5000
   - Check CORS configuration
   - Verify API key is valid

4. **Papers won't process**:
   - Check PDF file format
   - Ensure sufficient disk space
   - Verify Groq API key and quota

### Error Messages

- **"System not configured"**: Enter Groq API key in configuration
- **"No papers found"**: Upload or download papers first
- **"System not ready"**: Process papers before chatting
- **"Context overflow"**: Try shorter questions or clear chat history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Groq**: Fast AI inference API
- **LlamaIndex**: Document indexing and retrieval
- **Material-UI**: React components
- **arXiv**: Academic paper access

## 📞 Support

For issues and support:
1. Check the troubleshooting section
2. Review API documentation
3. Open an issue on GitHub

---

**Happy researching! 🔬📚**
