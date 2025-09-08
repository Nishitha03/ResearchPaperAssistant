import os
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import arxiv
import hashlib
import json
from typing import List, Dict, Optional

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

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = Path("./papers")
STORAGE_FOLDER = Path("./storage")
ALLOWED_EXTENSIONS = {'pdf'}

# Global variables
qa_system = None
groq_api_key = None

class AcademicPaperQA:
    def __init__(self, model_name="llama3-8b-8192", groq_api_key=None):
        self.data_dir = UPLOAD_FOLDER
        self.storage_dir = STORAGE_FOLDER
        self.model_name = model_name
        self.groq_api_key = groq_api_key
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self._setup_models()
        
        # Initialize components
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.current_papers_hash = None
        self.is_ready = False
        self.chat_history = []
    
    def _setup_models(self):
        try:
            if not self.groq_api_key:
                raise ValueError("Groq API key is required")
            
            # Initialize LLM
            self.llm = Groq(
                model=self.model_name,
                api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9,
                system_prompt="""You are an expert academic research assistant. Provide comprehensive, detailed responses about research papers including:

1. Direct answers to questions
2. Relevant background context
3. Specific details from papers including methodologies and findings
4. Analysis and interpretation
5. Connections between concepts when relevant

Keep responses thorough but concise to stay within token limits."""
            )
            
            # Initialize embedding model
            try:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    max_length=512
                )
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace embedding, trying alternative: {e}")
                self.embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    device="cpu",
                    max_length=512
                )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.chunk_size = 256
            Settings.chunk_overlap = 25
            
            logger.info(f"Models initialized successfully with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
    
    def _get_papers_hash(self) -> str:
        pdf_files = list(self.data_dir.glob("*.pdf"))
        if not pdf_files:
            return ""
        
        file_info = []
        for pdf_file in sorted(pdf_files):
            file_info.append(f"{pdf_file.name}:{pdf_file.stat().st_size}")
        
        papers_string = "|".join(file_info)
        return hashlib.md5(papers_string.encode()).hexdigest()
    
    def _save_papers_metadata(self, papers_hash: str):
        metadata_file = self.storage_dir / "papers_metadata.json"
        metadata = {
            "papers_hash": papers_hash,
            "model_name": self.model_name
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    
    def _load_papers_metadata(self) -> Dict:
        metadata_file = self.storage_dir / "papers_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        return {}
    
    def download_arxiv_paper(self, arxiv_id: str) -> Optional[str]:
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
        try:
            if file_paths is None:
                reader = SimpleDirectoryReader(
                    input_dir=str(self.data_dir),
                    required_exts=[".pdf"],
                    recursive=False
                )
            else:
                reader = SimpleDirectoryReader(input_files=file_paths)
            
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            # Clean and validate documents
            cleaned_documents = []
            for doc in documents:
                if doc.text and len(doc.text.strip()) > 50:
                    if len(doc.text) > 50000:
                        doc.text = doc.text[:50000] + "... [Document truncated]"
                    cleaned_documents.append(doc)
            
            logger.info(f"After cleaning: {len(cleaned_documents)} valid documents")
            return cleaned_documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def create_index(self, documents: List[Document], save_index: bool = True):
        try:
            if not documents:
                raise ValueError("No documents provided for indexing")
                
            logger.info(f"Creating index from {len(documents)} documents")
            
            sentence_splitter = SentenceSplitter(
                chunk_size=256,
                chunk_overlap=25,
                separator=" "
            )
            
            # Process documents in batches
            batch_size = 5
            all_nodes = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                nodes = sentence_splitter.get_nodes_from_documents(batch)
                all_nodes.extend(nodes)
            
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
        try:
            if not self.index:
                raise ValueError("No index available for query engine")
                
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=2
            )
            
            response_synthesizer = get_response_synthesizer(
                response_mode="compact",
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
        try:
            if not self.index:
                raise ValueError("No index available for chat engine")
            
            memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
            
            self.chat_engine = CondensePlusContextChatEngine.from_defaults(
                retriever=VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=2
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
                context_window=4096,
                max_tokens=1500
            )
            
            logger.info("Chat engine created successfully")
            
        except Exception as e:
            logger.error(f"Error creating chat engine: {e}")
            raise
    
    def get_loaded_papers_info(self) -> List[str]:
        pdf_files = list(self.data_dir.glob("*.pdf"))
        return [pdf_file.name for pdf_file in pdf_files]
    
    def clear_papers(self):
        try:
            for pdf_file in self.data_dir.glob("*.pdf"):
                pdf_file.unlink()
            
            if self.storage_dir.exists():
                import shutil
                shutil.rmtree(self.storage_dir)
                self.storage_dir.mkdir(exist_ok=True)
            
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
        try:
            self.chat_history = []
            if self.chat_engine and hasattr(self.chat_engine, 'memory'):
                self.chat_engine.memory.reset()
            logger.info("Chat history cleared")
        except Exception as e:
            logger.error(f"Error clearing chat history: {e}")
    
    def process_all_papers(self) -> Dict[str, str]:
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
        if not self.is_ready:
            return {"error": "System not ready. Please process papers first."}
        
        try:
            logger.info(f"Asking question: {question}")
            
            if len(question) > 500:
                question = question[:500] + "..."
                logger.warning("Question truncated to prevent context overflow")
            
            if use_chat_engine and self.chat_engine:
                try:
                    response = self.chat_engine.chat(question)
                    answer = str(response)
                except Exception as chat_error:
                    logger.warning(f"Chat engine failed, falling back to query engine: {chat_error}")
                    response = self.query_engine.query(question)
                    answer = str(response)
                    use_chat_engine = False
            else:
                response = self.query_engine.query(question)  
                answer = str(response)
            
            self.chat_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "answer": answer,
                "type": "chat" if use_chat_engine else "query"
            })
            
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/config', methods=['POST'])
def configure_api():
    global qa_system, groq_api_key
    
    data = request.get_json()
    groq_api_key = data.get('groq_api_key')
    model_name = data.get('model_name', 'llama3-8b-8192')
    
    if not groq_api_key:
        return jsonify({"error": "Groq API key is required"}), 400
    
    try:
        qa_system = AcademicPaperQA(model_name=model_name, groq_api_key=groq_api_key)
        return jsonify({"success": "System initialized successfully", "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    if not qa_system:
        return jsonify({"configured": False, "ready": False, "papers": []})
    
    return jsonify({
        "configured": True,
        "ready": qa_system.is_ready,
        "papers": qa_system.get_loaded_papers_info(),
        "chat_history": qa_system.chat_history[-10:]  # Last 10 messages
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if not qa_system:
        return jsonify({"error": "System not configured"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        return jsonify({"success": f"File {filename} uploaded successfully"})
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/download-arxiv', methods=['POST'])
def download_arxiv():
    if not qa_system:
        return jsonify({"error": "System not configured"}), 400
    
    data = request.get_json()
    arxiv_id = data.get('arxiv_id')
    
    if not arxiv_id:
        return jsonify({"error": "arXiv ID is required"}), 400
    
    filepath = qa_system.download_arxiv_paper(arxiv_id)
    if filepath:
        return jsonify({"success": "Paper downloaded successfully"})
    else:
        return jsonify({"error": "Failed to download paper"}), 500

@app.route('/api/process-papers', methods=['POST'])
def process_papers():
    if not qa_system:
        return jsonify({"error": "System not configured"}), 400
    
    result = qa_system.process_all_papers()
    
    if "error" in result:
        return jsonify(result), 500
    else:
        return jsonify(result)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    if not qa_system:
        return jsonify({"error": "System not configured"}), 400
    
    data = request.get_json()
    question = data.get('question')
    use_chat_engine = data.get('use_chat_engine', True)
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    result = qa_system.ask_question(question, use_chat_engine)
    
    if "error" in result:
        return jsonify(result), 500
    else:
        return jsonify(result)

@app.route('/api/clear-papers', methods=['POST'])
def clear_papers():
    if not qa_system:
        return jsonify({"error": "System not configured"}), 400
    
    if qa_system.clear_papers():
        return jsonify({"success": "Papers cleared successfully"})
    else:
        return jsonify({"error": "Failed to clear papers"}), 500

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    if not qa_system:
        return jsonify({"error": "System not configured"}), 400
    
    qa_system.clear_chat_history()
    return jsonify({"success": "Chat history cleared"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)