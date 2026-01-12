import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
import torch
import pynvml  # Add pynvml import
import time  # Add time import

# Add manually compiled llama-cpp-python to path if it exists
# Allows configuring the path via environment variable
_llama_cpp_path = os.getenv('LLAMA_CPP_PATH', '')
if _llama_cpp_path and os.path.exists(_llama_cpp_path) and _llama_cpp_path not in sys.path:
    sys.path.insert(0, _llama_cpp_path)

from config import SYSTEM_PROMPT, COMMAND_PREFIX, MODELS_DIR, MODEL_PATH, CHROMA_DB_PATH, MODEL_N_GPU_LAYERS, MEMORY_EXPIRY, MAX_HISTORY_LENGTH, CONTEXT_WINDOW_MESSAGES
from llm.model import GGUFModel
from rag.chroma_db import ChromaVectorStore
from rag.document import DocumentProcessor
from rag.retriever import RAGRetriever
from commands.parser import CommandParser
from commands.executor import CommandExecutor

# Configure logging
# Create console handler with UTF-8 (avoids errors on Windows with cp1252)
console_handler = logging.StreamHandler()
# Configure UTF-8 encoding for console handler on Windows
if sys.platform == 'win32':
    try:
        # Try to configure UTF-8 for stdout
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass  # If it fails, continue without reconfiguring

# Configure file handler with rotation (max 10MB per file, keep 5 backups)
file_handler = RotatingFileHandler(
    "rag_system.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,  # Keep 5 backup files
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        console_handler
    ]
)

logger = logging.getLogger(__name__)

# Log custom llama-cpp-python path if configured
if _llama_cpp_path:
    logger.info(f"Added custom llama-cpp-python path: {_llama_cpp_path}")

class RAGSystem:
    """Main class that integrates all RAG system components."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        db_path: Optional[str] = None,
        use_docker=True,
        docker_host="localhost",
        docker_port=8000,
        system_prompt: str = SYSTEM_PROMPT,
        force_cpu: bool = False
    ):
        """
        Initialize the RAG system.
        
        Args:
            model_path: Path to GGUF model (optional)
            db_path: Path to vector database (optional, not currently used)
            use_docker: Indicates if ChromaDB is used in Docker (optional, not currently used)
            docker_host: ChromaDB host (default: localhost)
            docker_port: ChromaDB port (default: 8000)
            system_prompt: System prompt
            force_cpu: Force CPU usage even if GPU is available
        """
        self.system_prompt = system_prompt
        
        # Check GPU with NVML
        gpu_available = False
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
            # Handle both bytes and strings (depending on pynvml version)
            if isinstance(gpu_name_raw, bytes):
                gpu_name = gpu_name_raw.decode('utf-8')
            else:
                gpu_name = str(gpu_name_raw)
            gpu_available = not force_cpu
            logger.info(f"GPU detected: {gpu_name}")
        except Exception as e:
            logger.warning(f"Could not initialize NVML: {e}")
            # Try to check CUDA with torch only if available, but don't depend on it
            try:
                gpu_available = torch.cuda.is_available() and not force_cpu if hasattr(torch, 'cuda') else False
            except:
                gpu_available = False
        
        # Use default model path if not provided
        if not model_path:
            if MODEL_PATH:
                model_path = str(MODEL_PATH)
                logger.info(f"Using default model: {model_path}")
            else:
                logger.warning("No model path provided and MODEL_NAME not configured in .env")
        
        # Initialize model
        self.llm = None
        if model_path:
            try:
                device_info = "GPU" if gpu_available else "CPU"
                logger.info(f"Loading model from {model_path} using {device_info}...")
                
                try:
                    self.llm = GGUFModel(
                        model_path=str(model_path),
                        n_gpu_layers=MODEL_N_GPU_LAYERS if gpu_available else 0
                    )
                    # Use model's GPU state
                    actual_device = "GPU" if self.llm.using_gpu else "CPU"
                    logger.info(f"Model loaded successfully on {actual_device}")
                except (RuntimeError, AttributeError) as gpu_error:
                    if gpu_available:
                        logger.warning(f"Error loading model on GPU: {str(gpu_error)}")
                        logger.info("Attempting to load model on CPU...")
                        self.llm = GGUFModel(
                            model_path=str(model_path),
                            n_gpu_layers=0
                        )
                        logger.info("Model loaded successfully on CPU")
                    else:
                        raise
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                logger.warning("System will run without language model")
        
        # Initialize RAG components
        # Use provided parameters or default values
        self.vector_store = ChromaVectorStore(
            host=docker_host, 
            port=docker_port,
            db_path=db_path
        )
        self.doc_processor = DocumentProcessor()
        self.rag_retriever = RAGRetriever(
            vector_store=self.vector_store,
            llm_model=self.llm,
            system_prompt=system_prompt
        )
        
        # Initialize command system
        self.command_parser = CommandParser()
        self.command_executor = CommandExecutor(
            command_parser=self.command_parser,
            document_processor=self.doc_processor,
            vector_store=self.vector_store,
            rag_retriever=self.rag_retriever,
            llm_model=self.llm,
            rag_system=self  # Pass self reference
        )
        
        # Initialize conversation memory
        self.conversation_history = []
        self.last_interaction_time = time.time()
        
        logger.info("RAG system initialized successfully")
    
    def _update_conversation_history(self, role: str, content: str) -> None:
        """Update conversation history."""
        current_time = time.time()
        
        # Reset history if too much time has passed
        if current_time - self.last_interaction_time > MEMORY_EXPIRY:
            self.conversation_history = []
            logger.info("Conversation memory reset due to inactivity")
        
        # Update last interaction time
        self.last_interaction_time = current_time
        
        # Add new message
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": current_time
        })
        
        # Keep only the last MAX_HISTORY_LENGTH messages
        if len(self.conversation_history) > MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-MAX_HISTORY_LENGTH:]
    
    def _get_conversation_context(self) -> str:
        """Get current conversation context."""
        if not self.conversation_history:
            return ""
            
        recent_messages = self.conversation_history[-CONTEXT_WINDOW_MESSAGES:]
        context = "\nConversation context:\n"
        for msg in recent_messages:
            context += f"{msg['role']}: {msg['content']}\n"
        return context
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input.
        
        Args:
            user_input: Text entered by the user
            
        Returns:
            Processing result
        """
        try:
            # Check if it's a command
            if self.command_parser.is_command(user_input):
                logger.info(f"Processing command: {user_input}")
                result = self.command_executor.execute_command(user_input)
                return result

            # Process as normal query
            logger.info(f"Processing query: {user_input}")
            
            # Update history with user input
            self._update_conversation_history("user", user_input)
            
            # Get conversation context
            conversation_context = self._get_conversation_context()
            
            # Perform RAG query with context
            enriched_query = f"{conversation_context}\nNew question: {user_input}"
            result = self.rag_retriever.query_with_rag(enriched_query)
            
            # Update history with response
            self._update_conversation_history("assistant", result["text"])
            
            return {
                "success": True,
                "message": result["text"],
                "is_command": False,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "is_command": False,
                "error": str(e)
            }
    
    def print_welcome(self) -> None:
        """Print welcome message."""
        print("\n" + "="*50)
        print("RAG System with GGUF Model")
        print("="*50)
        if self.llm:
            device_info = "GPU" if getattr(self.llm, "using_gpu", False) else "CPU"
            print(f"Model: Loaded ({device_info})")
        else:
            print("Model: Not available")
        
        # Check which mode ChromaDB is using
        if hasattr(self.vector_store, 'use_persistent') and self.vector_store.use_persistent:
            print(f"ChromaDB: Persistent mode at {self.vector_store.db_path}")
        else:
            host = getattr(self.vector_store, 'host', 'localhost')
            port = getattr(self.vector_store, 'port', 8000)
            print(f"ChromaDB: Server at {host}:{port}")
        
        print(f"Type '{COMMAND_PREFIX}help' to see available commands")
        print(f"Type 'exit' or 'quit' to exit")
        print(f"Type '{COMMAND_PREFIX}force_exit' to force system shutdown")
        print("="*50 + "\n")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG System with GGUF Model")
    parser.add_argument(
        "--model", "-m",
        help="Path to GGUF model",
        type=str,
        default=None
    )
    parser.add_argument(
        "--db", "-d",
        help="Path to vector database",
        type=str,
        default=None
    )
    parser.add_argument(
        "--force-cpu",
        help="Force CPU usage even if GPU is available",
        action="store_true"
    )
    args = parser.parse_args()

    # Initialize system
    system = RAGSystem(model_path=args.model, db_path=args.db, force_cpu=args.force_cpu)
    system.print_welcome()

    # Interactive loop
    while True:
        try:
            user_input = input(">>> ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("\nExiting system...")
                break

            result = system.process_input(user_input)
            print("\n" + "=" * 50)
            print(result["message"])
            print("=" * 50 + "\n")
            
            # Check if forced exit was requested
            if result.get("force_exit", False):
                break

        except KeyboardInterrupt:
            print("\n\nInterruption detected. Closing system...")
            break
        except EOFError:
            print("\n\nInput not available. Closing system...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print(f"Error: {str(e)}")
            break  # Exit loop to avoid infinite loops

if __name__ == "__main__":
    main()