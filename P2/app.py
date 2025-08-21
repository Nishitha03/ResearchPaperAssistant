# this code works perfectly with gemma 2 in ollama

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
from llama_index.llms.ollama import Ollama
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
    def __init__(self, model_name="gemma2:2b"):
        """Initialize the Academic Paper Q&A system"""
        self.data_dir = Path("./papers")
        self.storage_dir = Path("./storage")
        self.model_name = model_name
        
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
        """Setup LLM and embedding models with optimized settings for detailed responses"""
        try:
            # Initialize LLM via Ollama with settings optimized for detailed responses
            self.llm = Ollama(
                model=self.model_name,
                request_timeout=600.0,  # 10 minutes timeout for longer responses
                temperature=0.3,  # Slightly higher for more creative/detailed responses
                context_window=4096,  # Increased context window for better understanding
                num_predict=1024,  # Allow longer responses
                top_p=0.9,  # Nucleus sampling for better quality
                repeat_penalty=1.1,  # Prevent repetition
                system_prompt="""You are an expert academic research assistant. When answering questions about research papers, provide comprehensive, detailed responses that include:
                
1. Direct answers to the question asked
2. Relevant background context and explanations
3. Specific details from the papers including methodologies, findings, and implications
4. Analysis and interpretation of the information
5. Connections between different concepts when relevant
6. Limitations or caveats when appropriate

Always aim for thorough, well-structured responses that demonstrate deep understanding of the academic content. Use clear paragraphs and explain technical concepts when necessary."""
            )
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 1024  # Larger chunks for more context
            Settings.chunk_overlap = 100  # More overlap for continuity
            
            logger.info(f"Models initialized successfully with {self.model_name}")
            
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
        """Load documents from PDF files"""
        try:
            if file_paths is None:
                reader = SimpleDirectoryReader(
                    input_dir=str(self.data_dir),
                    required_exts=[".pdf"]
                )
            else:
                reader = SimpleDirectoryReader(input_files=file_paths)
            
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def create_index(self, documents: List[Document], save_index: bool = True):
        """Create vector index from documents"""
        try:
            if not documents:
                raise ValueError("No documents provided for indexing")
                
            logger.info(f"Creating index from {len(documents)} documents")
            
            sentence_splitter = SentenceSplitter(
                chunk_size=1024,  # Larger chunks
                chunk_overlap=100
            )
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                transformations=[sentence_splitter],
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
                similarity_top_k=5  # Increased for more comprehensive answers
            )
            
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",  # Better for detailed responses
                streaming=False,
                text_qa_template="""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, please provide a comprehensive and detailed answer to the question. Include specific details from the research papers, explain methodologies when relevant, discuss findings thoroughly, and provide analysis and implications. Structure your response clearly with proper explanations of technical concepts.

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
        """Create chat engine for conversational interactions"""
        try:
            if not self.index:
                raise ValueError("No index available for chat engine")
            
            # Create memory buffer for chat history
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            
            # Create chat engine
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=5
                ),
                memory=memory,
                llm=self.llm,
                context_prompt=(
                    "You are an expert academic research assistant having a conversation about research papers. "
                    "Use the following context from the papers to answer questions thoroughly and in detail. "
                    "Provide comprehensive explanations, include specific findings, methodologies, and implications. "
                    "Build upon previous parts of the conversation when relevant.\n"
                    "Context:\n"
                    "{context_str}\n"
                    "Instructions: Answer the user's question in detail using the provided context."
                ),
                verbose=True
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
            
            if use_chat_engine and self.chat_engine:
                # Use chat engine for conversational context
                response = self.chat_engine.chat(question)
                answer = str(response)
                
                # Add to chat history
                self.chat_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "question": question,
                    "answer": answer,
                    "type": "chat"
                })
                
                # Get sources if available
                sources = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for i, node in enumerate(response.source_nodes):
                        sources.append({
                            'text': node.text[:400] + "..." if len(node.text) > 400 else node.text,
                            'score': node.score if hasattr(node, 'score') else 'N/A'
                        })
                
            else:
                # Use query engine for standalone questions
                response = self.query_engine.query(question)
                answer = str(response)
                
                # Add to chat history
                self.chat_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "question": question,
                    "answer": answer,
                    "type": "query"
                })
                
                sources = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for i, node in enumerate(response.source_nodes):
                        sources.append({
                            'text': node.text[:400] + "..." if len(node.text) > 400 else node.text,
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
        page_title="Academic Paper Q&A Bot",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Academic Paper Q&A Bot")
    
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
    
    # Model selection in sidebar
    st.sidebar.header("⚙️ Configuration")
    model_options = {
        "Gemma 2B (Recommended)": "gemma2:2b",
        "Gemma 9B (Most Capable)": "gemma2:9b",
        "Llama3 8B (Balanced)": "llama3:8b",
        "Llama2 7B (Stable)": "llama2:7b"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model]
    
    # Initialize session state
    if 'qa_system' not in st.session_state or st.session_state.get('current_model') != model_name:
        with st.spinner(f"Initializing system with {selected_model}..."):
            try:
                st.session_state.qa_system = AcademicPaperQA(model_name=model_name)
                st.session_state.current_model = model_name
                st.session_state.papers_loaded = False
                st.success(f"System initialized with {selected_model}!")
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                st.info("Make sure Ollama is running and the model is installed.")
                st.stop()
    
    if 'papers_loaded' not in st.session_state:
        st.session_state.papers_loaded = False
    
    # Display current model info
    st.sidebar.info(f"**Current model:** {selected_model}")
    
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
    
    # Response mode toggle
    use_chat_mode = st.sidebar.toggle("💬 Conversational Mode", value=True, 
                                     help="Enable for follow-up questions and context retention")
    
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
                with st.spinner("Processing papers..."):
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
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-content"><strong style ="color: black;">You:</strong> {message['question']}</div>
                    <div class="message-timestamp">{message['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-content"><strong style ="color: black;">Assistant:</strong></div>
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
                quick_question = "What is the main research question or objective addressed in this paper? Please provide a detailed explanation."
            if st.button("🔬 Methodology"):
                quick_question = "What methodology or research approach was used in this study? Please explain in detail including any experimental design, data collection methods, and analytical techniques."
        
        with col2:
            if st.button("📊 Key Findings"):
                quick_question = "What are the key findings and results of this research? Please provide a comprehensive summary of the main discoveries and their significance."
            if st.button("🎯 Conclusions"):
                quick_question = "What are the main conclusions and implications of this research? How do the authors interpret their findings?"
        
        with col3:
            if st.button("⚠️ Limitations"):
                quick_question = "What are the limitations of this study? What do the authors identify as potential weaknesses or areas for future research?"
            if st.button("📋 Summary"):
                quick_question = "Please provide a comprehensive summary of this paper including the research question, methodology, key findings, and conclusions."
        
        # Chat input
        st.subheader("💭 Ask Your Question")
        user_question = st.text_area("Type your question here...", height=100, placeholder="Ask anything about your papers...")
        
        # Use quick question if selected, otherwise use user input
        question_to_ask = quick_question if quick_question else user_question
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Send Message", type="primary", disabled=not question_to_ask):
                if question_to_ask:
                    with st.spinner("Thinking..."):
                        result = st.session_state.qa_system.ask_question(
                            question_to_ask, 
                            use_chat_engine=use_chat_mode
                        )
                        
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.rerun()  # Reload to show new message
        
        with col2:
            response_mode = "💬 Chat Mode" if use_chat_mode else "❓ Q&A Mode"
            st.info(response_mode)
        
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



