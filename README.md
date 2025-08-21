# Research Paper Assistant 🔬

An intelligent academic paper Q&A system powered by Groq API that allows you to upload research papers and interact with them through natural language queries. Built with LlamaIndex, Streamlit, and Groq for fast cloud-based LLM inference.

## Features ✨

- **Multi-format Paper Loading**: Upload PDFs or download directly from arXiv
- **Intelligent Q&A**: Ask detailed questions about your research papers
- **Conversational Interface**: Chat-like experience with context retention
- **Source Citations**: View relevant excerpts from papers that informed each answer
- **Multiple LLM Support**: Compatible with various Groq models (Llama3, Mixtral, Gemma)
- **Persistent Storage**: Index caching for faster subsequent queries
- **Quick Questions**: Pre-built buttons for common research queries
- **Cloud-Based**: No local GPU required - powered by Groq's fast inference API

## Installation 🚀

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ResearchPaperAssistant.git
   cd ResearchPaperAssistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r P2/requirements.txt
   ```

3. **Get Groq API Key** (Free)
   - Visit [console.groq.com/keys](https://console.groq.com/keys)
   - Create a free account
   - Generate an API key
   - Either enter it in the app sidebar or set as environment variable: `GROQ_API_KEY`

## Usage 📖

1. **Start the application**
   ```bash
   cd P2
   streamlit run app1.py
   ```

2. **Configure your API key**
   - Enter your Groq API key in the sidebar
   - Select from available Groq models (Llama3 8B recommended for stability)

3. **Load research papers**
   - Upload PDF files directly, or
   - Enter arXiv IDs to download papers automatically

4. **Process papers**
   - Click "Process Papers" to create searchable index
   - Embeddings are generated locally (CPU-based)

5. **Start asking questions**
   - Use quick question buttons for common queries
   - Type custom questions in the chat interface
   - View sources and citations for each answer

## Quick Questions 🚀

The app includes built-in quick questions for efficient research analysis:

- **Main Research Question**: What problem does this paper address?
- **Methodology**: How was the research conducted?
- **Key Findings**: What are the main results and discoveries?
- **Conclusions**: What are the implications and interpretations?
- **Limitations**: What are the study's weaknesses or constraints?
- **Summary**: Comprehensive overview of the entire paper

## Supported Models 🤖

- **Llama3 8B** (Recommended) - Fast and stable for most use cases
- **Llama3 70B** - Most capable but slower
- **Mixtral 8x7B** - Balanced performance and capability
- **Gemma 7B** - Efficient and lightweight

## Requirements 📋

- Python 3.8+
- Free Groq API key
- 4GB+ RAM (for local embeddings)
- Internet connection (for Groq API)

## File Structure 📁

```
ResearchPaperAssistant/
├── P2/
│   ├── app.py              # Main Streamlit application
│   ├── app1.py             # Alternative implementation
│   ├── requirements.txt    # Python dependencies
│   ├── test.py            # Test script
│   ├── papers/            # Uploaded/downloaded papers (created at runtime)
│   ├── storage/           # Vector index storage (created at runtime)
│   └── embeddings_cache/  # Cached embedding models
└── README.md
```

## Configuration ⚙️

### Model Selection
Choose between different Ollama models based on your hardware:
- **2-4GB RAM**: Use Gemma 2B or TinyLlama
- **8GB+ RAM**: Use Gemma 9B or Llama3 8B
- **16GB+ RAM**: Any model will work well

### Performance Tips
- Use smaller, quantized models for faster responses
- Process papers in smaller batches if memory is limited
- Enable conversational mode for context-aware follow-up questions

## API Features 🔧

### Core Functions
- `download_arxiv_paper(arxiv_id)` - Download papers from arXiv
- `load_documents()` - Process uploaded PDF files
- `create_index()` - Build searchable vector index
- `ask_question()` - Query the knowledge base
- `clear_papers()` - Reset and start fresh

### Advanced Features
- Smart index caching and rebuilding
- Conversation memory management
- Source tracking and citation
- Batch document processing

## Troubleshooting 🔧

**Common Issues:**

1. **Ollama not found**: Ensure Ollama is installed and running
2. **Model not available**: Pull the model first: `ollama pull model-name`
3. **Out of memory**: Use a smaller model or reduce chunk size
4. **Slow processing**: Consider using quantized models (Q4_0, Q3_K_M)

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License 📄

This project is open source and available under the MIT License.

## Acknowledgments 🙏

- **LlamaIndex** for the RAG framework
- **Streamlit** for the web interface
- **Ollama** for local LLM inference
- **Hugging Face** for embedding models
- **arXiv** for academic paper access
