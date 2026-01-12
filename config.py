# config.py
"""
Centralized configuration for RAG system with GGUF model.
Uses environment variables from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', './models'))

# Model names (automatically combined with MODELS_DIR)
# If MODEL_NAME is just a filename, it's combined with MODELS_DIR
_model_name = os.getenv('MODEL_NAME', '').strip()
if _model_name:
    _model_path = Path(_model_name)
    # Normalize path separators
    _model_path_str = str(_model_path).replace('\\', '/')
    _model_path = Path(_model_path_str)
    
    # If it's an absolute path, use it directly
    if _model_path.is_absolute():
        MODEL_PATH = _model_path
    # If it starts with ./ or .\, it's relative to current directory
    elif str(_model_path).startswith('./') or str(_model_path).startswith('.\\'):
        MODEL_PATH = BASE_DIR / _model_path
    # If it's just a filename or relative path, combine with MODELS_DIR
    else:
        # Check if file exists directly
        if _model_path.exists():
            MODEL_PATH = _model_path
        # If it doesn't exist, try in MODELS_DIR
        elif (MODELS_DIR / _model_path).exists():
            MODEL_PATH = MODELS_DIR / _model_path
        # If it doesn't exist there either, assume it's relative to MODELS_DIR
        else:
            MODEL_PATH = MODELS_DIR / _model_path
else:
    # If MODEL_NAME is not configured, automatically search for GGUF models in MODELS_DIR
    MODEL_PATH = None
    if MODELS_DIR.exists():
        # Search for .gguf files that are not embeddings
        gguf_files = list(MODELS_DIR.glob("*.gguf"))
        # Filter out common embedding models
        embedding_keywords = ['embed', 'embedding', 'nomic']
        main_models = [f for f in gguf_files if not any(kw in f.name.lower() for kw in embedding_keywords)]
        if main_models:
            # Use the first model found (or the most recent)
            MODEL_PATH = sorted(main_models, key=lambda x: x.stat().st_mtime, reverse=True)[0]

# Similar for embedding model
# Try GGUF_EMBEDDING_MODEL_PATH first, then fallback to EMBEDDING_MODEL_NAME for backwards compatibility
_embedding_model_path = os.getenv('GGUF_EMBEDDING_MODEL_PATH', '')
if not _embedding_model_path:
    _embedding_model_path = os.getenv('EMBEDDING_MODEL_NAME', '')

if _embedding_model_path:
    _embedding_path = Path(_embedding_model_path)
    # Normalize path separators
    _embedding_path_str = str(_embedding_path).replace('\\', '/')
    _embedding_path = Path(_embedding_path_str)
    
    # If it's an absolute path, use it directly
    if _embedding_path.is_absolute():
        EMBEDDING_MODEL_PATH = _embedding_path
    # If it starts with ./ or .\, it's relative to current directory
    elif str(_embedding_path).startswith('./') or str(_embedding_path).startswith('.\\'):
        EMBEDDING_MODEL_PATH = BASE_DIR / _embedding_path
    # If it's just a filename or relative path, combine with MODELS_DIR
    else:
        # Check if file exists directly
        if _embedding_path.exists():
            EMBEDDING_MODEL_PATH = _embedding_path
        # If it doesn't exist, try in MODELS_DIR
        elif (MODELS_DIR / _embedding_path).exists():
            EMBEDDING_MODEL_PATH = MODELS_DIR / _embedding_path
        # If it doesn't exist there either, assume it's relative to MODELS_DIR
        else:
            EMBEDDING_MODEL_PATH = MODELS_DIR / _embedding_path
else:
    EMBEDDING_MODEL_PATH = None

# Keep backwards compatibility (alias)
GGUF_EMBEDDING_MODEL_PATH = EMBEDDING_MODEL_PATH

# Embedding model configuration
EMBEDDING_MODEL_CONTEXT_SIZE = int(os.getenv('EMBEDDING_MODEL_CONTEXT_SIZE', 2048))
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 384))
EMBEDDING_MODEL_N_GPU_LAYERS = int(os.getenv('EMBEDDING_MODEL_N_GPU_LAYERS', -1))
EMBEDDING_MODEL_N_THREADS = int(os.getenv('EMBEDDING_MODEL_N_THREADS', 4))

# ChromaDB configuration
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', 8000))
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'documents')

# GPU configuration
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', 0.7))
GPU_MAIN_DEVICE = int(os.getenv('GPU_MAIN_DEVICE', 0))
N_BATCH = int(os.getenv('N_BATCH', 512))
MODEL_N_GPU_LAYERS = int(os.getenv('MODEL_N_GPU_LAYERS', -1))  # Renamed from N_GPU_LAYERS
N_GPU_LAYERS = MODEL_N_GPU_LAYERS  # Keep backwards compatibility

# Main model configuration
MODEL_CONTEXT_SIZE = int(os.getenv('MODEL_CONTEXT_SIZE', 4096))
MODEL_N_THREADS = int(os.getenv('MODEL_N_THREADS', 4))
MODEL_SEED = int(os.getenv('MODEL_SEED', 42))

# Generation configuration
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
TOP_P = float(os.getenv('TOP_P', 0.9))
TOP_K = int(os.getenv('TOP_K', 40))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 2048))
REPEAT_PENALTY = float(os.getenv('REPEAT_PENALTY', 1.1))

# RAG parameters
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))
NUM_RETRIEVAL_RESULTS = int(os.getenv('NUM_RETRIEVAL_RESULTS', 5))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.2))
MAX_RELEVANT_CHUNKS = int(os.getenv('MAX_RELEVANT_CHUNKS', 5))

# Memory configuration
MAX_HISTORY_LENGTH = int(os.getenv('MAX_HISTORY_LENGTH', 10))
CONTEXT_WINDOW_MESSAGES = int(os.getenv('CONTEXT_WINDOW_MESSAGES', 5))
MEMORY_EXPIRY = int(os.getenv('MEMORY_EXPIRY', 3600))

# Prefixes and commands (keep fixed values)
COMMAND_PREFIX = "/"

# Available commands list
AVAILABLE_COMMANDS = {
    "help": "Show help",
    "search": "Search for specific documents",
    "add": "Add documents to the system",
    "delete": "Delete documents",
    "list": "List documents",
    "settings": "Modify configuration",
    "force_exit": "Force system shutdown",
    "history": "View conversation history",
    "clear": "Clear history",
    "context": "Manage conversation context"
}

# System messages
SYSTEM_PROMPT = """You are a helpful AI assistant named "TARS" that answers questions based on information from available documents.
If you don't know the answer, say so clearly instead of inventing information.
You can use commands with the / prefix to perform specific actions. You are sometimes sarcastic, but you also like to be funny when possible, without losing accuracy and a pragmatic, scientific approach.
"""