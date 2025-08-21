
import os
import requests
import arxiv
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings,
    Document,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import logging
import hashlib
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AcademicPaperQA:
    def __init__(self, model_name="llama3-70b-8192", groq_api_key=None):
        """Initialize the Academic Paper Q&A system with Groq API"""
        self.data_dir = Path("./papers")
        self.storage_dir = Path("./storage")
        self.model_name = model_name
        self.groq_api_key = groq_api_key
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self._setup_models()
        
        # Initialize index and chat engine
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.current_papers_hash = None
        self.is_ready = False
        
        # Chat history
        self.chat_history = []
    
    def _setup_models(self):
        """Setup LLM and embedding models with Groq API"""
        try:
            if not self.groq_api_key:
                raise ValueError("Groq API key is required. Please set GROQ_API_KEY environment variable or pass it directly.")
            
            # Initialize LLM via Groq API with conservative token settings
            self.llm = Groq(
                model=self.model_name,
                api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=2048,  # Reduced max tokens to prevent context overflow
                top_p=0.9,
                system_prompt="""You are an expert academic research assistant. Provide comprehensive, detailed responses about research papers including:

1. Direct answers to questions
2. Relevant background context
3. Specific details from papers including methodologies and findings
4. Analysis and interpretation
5. Connections between concepts when relevant

Keep responses thorough but concise to stay within token limits."""
            )
            
            # Initialize lightweight embedding model for CPU usage
            # Using a more stable embedding model
            try:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    max_length=512  # Explicit max length to prevent issues
                )
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace embedding, trying alternative: {e}")
                # Fallback to a different embedding model
                self.embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    device="cpu",
                    max_length=512
                )
            
            # Configure global settings with conservative values
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 256  # Smaller chunks to prevent context overflow
            Settings.chunk_overlap = 25  # Reduced overlap
            
            logger.info(f"Models initialized successfully with {self.model_name} via Groq API")
            
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
    
    def _get_papers_hash(self) -> str:
        """Generate hash of current papers in directory"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            return ""
        
        # Create hash based on filenames and file sizes
        file_info = []
        for pdf_file in sorted(pdf_files):
            file_info.append(f"{pdf_file.name}:{pdf_file.stat().st_size}")
        
        papers_string = "|".join(file_info)
        return hashlib.md5(papers_string.encode()).hexdigest()
    
    def _save_papers_metadata(self, papers_hash: str):
        """Save metadata about current papers"""
        metadata_file = self.storage_dir / "papers_metadata.json"
        metadata = {
            "papers_hash": papers_hash,
            "model_name": self.model_name
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    
    def _load_papers_metadata(self) -> Dict:
        """Load metadata about papers"""
        metadata_file = self.storage_dir / "papers_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def download_arxiv_paper(self, arxiv_id: str) -> Optional[str]:
        """Download paper from arXiv"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            filename = f"{arxiv_id.replace('/', '_')}.pdf"
            filepath = self.data_dir / filename
            
            paper.download_pdf(dirpath=str(self.data_dir), filename=filename)
            
            logger.info(f"Downloaded paper: {paper.title}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading paper {arxiv_id}: {e}")
            return None
    
    def load_documents(self, file_paths: List[str] = None) -> List[Document]:
        """Load documents from PDF files with error handling"""
        try:
            if file_paths is None:
                reader = SimpleDirectoryReader(
                    input_dir=str(self.data_dir),
                    required_exts=[".pdf"],
                    recursive=False  # Explicit setting
                )
            else:
                reader = SimpleDirectoryReader(input_files=file_paths)
            
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            # Clean and validate documents
            cleaned_documents = []
            for doc in documents:
                if doc.text and len(doc.text.strip()) > 50:  # Filter out very short documents
                    # Truncate very long documents to prevent memory issues
                    if len(doc.text) > 50000:
                        doc.text = doc.text[:50000] + "... [Document truncated]"
                    cleaned_documents.append(doc)
            
            logger.info(f"After cleaning: {len(cleaned_documents)} valid documents")
            return cleaned_documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def create_index(self, documents: List[Document], save_index: bool = True):
        """Create vector index from documents with CPU-optimized settings"""
        try:
            if not documents:
                raise ValueError("No documents provided for indexing")
                
            logger.info(f"Creating index from {len(documents)} documents")
            
            # CPU-optimized sentence splitter with smaller chunks
            sentence_splitter = SentenceSplitter(
                chunk_size=256,  # Smaller chunks to prevent context overflow
                chunk_overlap=25,
                separator=" "  # Explicit separator
            )
            
            # Process documents in smaller batches to prevent memory issues
            batch_size = 5
            all_nodes = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                nodes = sentence_splitter.get_nodes_from_documents(batch)
                all_nodes.extend(nodes)
            
            # Create index from nodes
            self.index = VectorStoreIndex(
                nodes=all_nodes,
                show_progress=True
            )
            
            if save_index:
                self.index.storage_context.persist(persist_dir=str(self.storage_dir))
                current_hash = self._get_papers_hash()
                self._save_papers_metadata(current_hash)
                self.current_papers_hash = current_hash
                logger.info("Index saved to storage")
            
            self._create_query_engine()
            self._create_chat_engine()
            self.is_ready = True
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            self.is_ready = False
            raise
    
    def should_rebuild_index(self) -> bool:
        """Check if index should be rebuilt based on papers"""
        current_hash = self._get_papers_hash()
        
        if not current_hash:
            return False
            
        metadata = self._load_papers_metadata()
        
        if not metadata:
            logger.info("No metadata found, rebuilding index")
            return True
        
        if metadata.get("papers_hash") != current_hash:
            logger.info("Papers hash changed, rebuilding index")
            return True
            
        if metadata.get("model_name") != self.model_name:
            logger.info("Model changed, rebuilding index")
            return True
            
        return False
    
    def load_index(self) -> bool:
        """Load existing index from storage if it matches current papers"""
        try:
            if self.should_rebuild_index():
                logger.info("Index needs to be rebuilt due to changes")
                return False
            
            index_files = list(self.storage_dir.glob("*"))
            if not index_files:
                logger.info("No index files found")
                return False
            
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.storage_dir)
            )
            self.index = load_index_from_storage(storage_context)
            self._create_query_engine()
            self._create_chat_engine()
            self.current_papers_hash = self._get_papers_hash()
            self.is_ready = True
            
            logger.info("Index loaded from storage successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.is_ready = False
            return False
    
    def _create_query_engine(self):
        """Create query engine with settings for detailed responses"""
        try:
            if not self.index:
                raise ValueError("No index available for query engine")
                
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=2  # Reduced to prevent context overflow
            )
            
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",  # More efficient for context management
                streaming=False,
                text_qa_template="""Context information is below.
---------------------
{context_str}
---------------------
Based on the context information, provide a comprehensive answer to the question. Include specific details from the research papers and explain key concepts clearly.

Question: {query_str}
Answer: """
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            logger.info("Query engine created successfully")
            
        except Exception as e:
            logger.error(f"Error creating query engine: {e}")
            raise
    
    def _create_chat_engine(self):
        """Create chat engine for conversational interactions with conservative settings"""
        try:
            if not self.index:
                raise ValueError("No index available for chat engine")
            
            # Create memory buffer with smaller token limit to prevent overflow
            memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
            
            # Create chat engine with conservative settings
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=2  # Reduced to manage context size
                ),
                memory=memory,
                llm=self.llm,
                context_prompt=(
                    "You are an expert academic research assistant. "
                    "Use the following context to answer questions thoroughly but concisely. "
                    "Context:\n{context_str}\n"
                    "Answer the user's question based on the provided context."
                ),
                verbose=True,
                # Additional context management
                context_window=4096,  # Conservative context window
                max_tokens=1500  # Conservative max tokens for response
            )
            
            logger.info("Chat engine created successfully")
            
        except Exception as e:
            logger.error(f"Error creating chat engine: {e}")
            raise
    
    def get_loaded_papers_info(self) -> List[str]:
        """Get list of currently loaded papers"""
        pdf_files = list(self.data_dir.glob("*.pdf"))
        return [pdf_file.name for pdf_file in pdf_files]
    
    def clear_papers(self):
        """Clear all papers and reset index"""
        try:
            # Remove all PDF files
            for pdf_file in self.data_dir.glob("*.pdf"):
                pdf_file.unlink()
            
            # Clear storage
            if self.storage_dir.exists():
                import shutil
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(exist_ok=True)
            
            # Reset everything
            self.index = None
            self.query_engine = None
            self.chat_engine = None
            self.current_papers_hash = None
            self.is_ready = False
            self.chat_history = []
            
            logger.info("Papers and index cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing papers: {e}")
            return False
    
    def clear_chat_history(self):
        """Clear chat history and reset memory"""
        try:
            self.chat_history = []
            if self.chat_engine and hasattr(self.chat_engine, 'memory'):
                self.chat_engine.memory.reset()
            logger.info("Chat history cleared")
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
    
    def process_all_papers(self) -> Dict[str, str]:
        """Process all papers in the directory and create/load index"""
        try:
            current_papers = self.get_loaded_papers_info()
            if not current_papers:
                return {"error": "No papers found in directory"}
            
            logger.info(f"Processing {len(current_papers)} papers: {current_papers}")
            
            if self.load_index():
                return {"success": f"Loaded existing index for {len(current_papers)} papers"}
            
            logger.info("Creating new index from documents...")
            documents = self.load_documents()
            
            if not documents:
                return {"error": "Failed to load documents from PDF files"}
            
            self.create_index(documents)
            
            if self.is_ready:
                return {"success": f"Successfully created index for {len(current_papers)} papers"}
            else:
                return {"error": "Failed to create index"}
                
        except Exception as e:
            logger.error(f"Error processing papers: {e}")
            return {"error": f"Error processing papers: {str(e)}"}
    
    def ask_question(self, question: str, use_chat_engine: bool = True) -> Dict[str, any]:
        """Ask a question using either chat engine (conversational) or query engine (standalone)"""
        if not self.is_ready:
            return {"error": "System not ready. Please process papers first."}
        
        try:
            logger.info(f"Asking question: {question}")
            
            # Truncate very long questions to prevent context overflow
            if len(question) > 500:
                question = question[:500] + "..."
                logger.warning("Question truncated to prevent context overflow")
            
            if use_chat_engine and self.chat_engine:
                # Use chat engine for conversational context
                try:
                    response = self.chat_engine.chat(question)
                    answer = str(response)
                except Exception as chat_error:
                    logger.warning(f"Chat engine failed, falling back to query engine: {chat_error}")
                    # Fallback to query engine if chat engine fails
                    response = self.query_engine.query(question)
                    answer = str(response)
                    use_chat_engine = False  # Update flag for history tracking
                
            else:
                # Use query engine for standalone questions
                response = self.query_engine.query(question)  
                answer = str(response)
            
            # Add to chat history
            self.chat_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "answer": answer,
                "type": "chat" if use_chat_engine else "query"
            })
            
            # Get sources if available
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    sources.append({
                        'text': node.text[:300] + "..." if len(node.text) > 300 else node.text,
                        'score': node.score if hasattr(node, 'score') else 'N/A'
                    })
            
            logger.info(f"Generated answer length: {len(answer)} characters")
            
            return {
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"error": f"Error processing question: {str(e)}"}

def create_streamlit_app():
    """Create Streamlit web interface with chat functionality"""
    st.set_page_config(
        page_title="Academic Paper Q&A Bot (Groq Powered)",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Academic Paper Q&A Bot (Groq Powered)")
    
    # Custom CSS for chat interface
    st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-content {
        margin: 0.5rem 0;
    }
    .message-timestamp {
        font-size: 0.8rem;
        color: #666;
        align-self: flex-end;
    }
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # API Key configuration in sidebar
    st.sidebar.header("🔑 API Configuration")
    groq_api_key = st.sidebar.text_input(
        "Groq API Key:",
        type="password",
        help="Get your free API key from https://console.groq.com/keys"
    )
    
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.sidebar.error("Please enter your Groq API key or set GROQ_API_KEY environment variable")
        st.info("🔑 **To get started:**\n1. Go to https://console.groq.com/keys\n2. Create a free account\n3. Generate an API key\n4. Enter it in the sidebar")
        st.stop()
    
    # Model selection in sidebar
    st.sidebar.header("⚙️ Configuration")
    model_options = {
        "Llama3 8B (Fast & Stable)": "llama3-8b-8192", 
        "Llama3 70B (Most Capable)": "llama3-70b-8192",
        "Mixtral 8x7B (Balanced)": "mixtral-8x7b-32768",
        "Gemma 7B (Efficient)": "gemma-7b-it"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Groq Model:",
        list(model_options.keys()),
        index=0  # Default to the more stable 8B model
    )
    
    model_name = model_options[selected_model]
    
    # Initialize session state
    if ('qa_system' not in st.session_state or 
        st.session_state.get('current_model') != model_name or
        st.session_state.get('current_api_key') != groq_api_key):
        
        with st.spinner(f"Initializing system with {selected_model}..."):
            try:
                st.session_state.qa_system = AcademicPaperQA(
                    model_name=model_name, 
                    groq_api_key=groq_api_key
                )
                st.session_state.current_model = model_name
                st.session_state.current_api_key = groq_api_key
                st.session_state.papers_loaded = False
                st.success(f"System initialized with {selected_model} via Groq API!")
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                st.info("Please check your Groq API key and try again.")
                st.stop()
    
    if 'papers_loaded' not in st.session_state:
        st.session_state.papers_loaded = False
    
    # Display current model info
    st.sidebar.info(f"**Current model:** {selected_model}")
    st.sidebar.success("✅ Using Groq API (Cloud)")
    st.sidebar.info("💬 Conversational Mode: ON")
    
    # Show system status
    if hasattr(st.session_state.qa_system, 'is_ready'):
        if st.session_state.qa_system.is_ready:
            st.sidebar.success("✅ System Ready")
        else:
            st.sidebar.warning("⚠️ Process papers first")
    
    # Show currently loaded papers
    current_papers = st.session_state.qa_system.get_loaded_papers_info()
    if current_papers:
        st.sidebar.subheader("📚 Loaded Papers:")
        for paper in current_papers:
            st.sidebar.text(f"📄 {paper}")
        
        if st.sidebar.button("🗑️ Clear All Papers"):
            with st.spinner("Clearing papers..."):
                if st.session_state.qa_system.clear_papers():
                    st.session_state.papers_loaded = False
                    st.sidebar.success("Papers cleared!")
                    st.rerun()
    
    # Chat controls in sidebar
    st.sidebar.subheader("💬 Chat Controls")
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.qa_system.clear_chat_history()
        st.sidebar.success("Chat cleared!")
        st.rerun()
    
    # Main interface
    if not st.session_state.qa_system.is_ready:
        # Show paper loading interface when system not ready
        st.header("📥 Load Academic Papers")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("From arXiv")
            arxiv_id = st.text_input("Enter arXiv ID (e.g., 2301.00001)")
            if st.button("Download from arXiv"):
                if arxiv_id:
                    with st.spinner("Downloading paper..."):
                        filepath = st.session_state.qa_system.download_arxiv_paper(arxiv_id)
                        if filepath:
                            st.success(f"Downloaded paper")
                            st.session_state.papers_loaded = False
                        else:
                            st.error("Failed to download paper")
        
        with col2:
            st.subheader("Upload PDF Files")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                saved_files = []
                for uploaded_file in uploaded_files:
                    file_path = st.session_state.qa_system.data_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(str(file_path))
                
                st.success(f"Uploaded {len(saved_files)} files")
                st.session_state.papers_loaded = False
        
        # Process papers
        st.subheader("🔄 Process Papers")
        current_papers = st.session_state.qa_system.get_loaded_papers_info()
        
        if not current_papers:
            st.info("No papers found. Please upload or download papers first.")
        else:
            st.info(f"Found {len(current_papers)} paper(s): {', '.join(current_papers)}")
            
            if st.button("🚀 Process Papers", type="primary"):
                with st.spinner("Processing papers (creating embeddings on CPU)..."):
                    result = st.session_state.qa_system.process_all_papers()
                    
                    if "error" in result:
                        st.error(result["error"])
                        st.session_state.papers_loaded = False
                    else:
                        st.success(result["success"])
                        st.session_state.papers_loaded = True
                        st.rerun()
    
    else:
        # Main chat interface when system is ready
        st.header("💬 Chat with Your Papers")
        
        # Show loaded papers info
        loaded_papers = st.session_state.qa_system.get_loaded_papers_info()
        st.info(f"📚 Chatting with {len(loaded_papers)} paper(s): {', '.join(loaded_papers)}")
        
        # Chat history display
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, message in enumerate(st.session_state.qa_system.chat_history[-10:]):  # Show last 10 messages
                

                st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-content" style="color: black;">
                            <strong>You:</strong> {message['question']}
                        </div>
                        <div class="message-timestamp">{message['timestamp']}</div>
                    </div>
                    """, unsafe_allow_html=True)


                
                # Bot response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-content"><strong style="color: black;">Assistant:</strong></div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(message['answer'])
                st.markdown("---")
        
        # Quick question buttons
        st.subheader("🚀 Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        quick_question = None
        with col1:
            if st.button("🎯 Main Research Question"):
                quick_question = "What is the main research question addressed in this paper?"
            if st.button("🔬 Methodology"):
                quick_question = "What methodology was used in this study?"
        
        with col2:
            if st.button("📊 Key Findings"):
                quick_question = "What are the key findings of this research?"
            if st.button("🎯 Conclusions"):
                quick_question = "What are the main conclusions of this research?"
        
        with col3:
            if st.button("⚠️ Limitations"):
                quick_question = "What are the limitations of this study?"
            if st.button("📋 Summary"):
                quick_question = "Please provide a summary of this paper."
        
        # Chat input
        st.subheader("💭 Ask Your Question")
        user_question = st.text_area("Type your question here...", height=100, placeholder="Ask anything about your papers...")
        
        # Use quick question if selected, otherwise use user input
        question_to_ask = quick_question if quick_question else user_question
        
        if st.button("Send Message", type="primary", disabled=not question_to_ask):
            if question_to_ask:
                with st.spinner("Thinking... (Processing via Groq API)"):
                    result = st.session_state.qa_system.ask_question(
                        question_to_ask, 
                        use_chat_engine=True  # Always use conversational mode
                    )
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.rerun()  # Reload to show new message
        
        # Sources section (show for last question if available)
        if (st.session_state.qa_system.chat_history and 
            st.session_state.qa_system.chat_history[-1].get('sources')):
            
            with st.expander("📚 View Sources", expanded=False):
                sources = st.session_state.qa_system.chat_history[-1]['sources']
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}** (Relevance: {source['score']})")
                    st.text(source['text'])
                    st.markdown("---")

if __name__ == "__main__":
    create_streamlit_app()
