# ATLAS-AI: Local RAG System with GPU Acceleration

This system implements a RAG (Retrieval Augmented Generation) that uses GGUF models locally with GPU acceleration through CUDA.

## 🌟 Features

- **Local Model**: Uses GGUF models for local inference with GPU acceleration
- **Efficient Embeddings**: Embedding generation with GPU optimization
- **Vector Database**: ChromaDB for efficient storage and search
- **Document Processing**: Support for multiple formats (PDF, DOCX, TXT, MD, HTML)
- **Command System**: Intuitive CLI interface with slash (/) commands
- **Conversational Memory**: Maintains context from previous conversations
- **GPU Monitoring**: VRAM usage tracking and resource optimization

## 🛠️ System Requirements

- Python 3.10 or higher
- CUDA Toolkit 12.x
- Minimum 8GB VRAM (recommended)
- Docker (ideal for running ChromaDB)

## ⚡ Installation

1. **Clone repository and create virtual environment**:
```bash
git clone "https://github.com/Silverx1242/ATLAS-AI.git"
cd ATLAS-AI
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. **Install CUDA and necessary dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev
```

3. **Install llama-cpp-python with CUDA support**:
   ```bash
   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
   ```

4. **Install project dependencies**:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
ATLAS-AI/
├── llm/                    # Model handling modules
│   ├── model.py           # GGUF model implementation
│   └── prompt.py          # Prompt handling
├── rag/                   # RAG components
│   ├── chroma_db.py       # ChromaDB integration
│   ├── document.py        # Document processing
│   └── retriever.py       # Information retrieval
├── commands/              # Command system
│   ├── parser.py         # Command parser
│   └── executor.py       # Command executor
├── config.py             # Centralized configuration
├── main.py              # Entry point
├── .env.example         # Environment variables example
└── requirements.txt     # Python dependencies
```

## ⚙️ Configuration

1. **Environment Variables**:
Copy the `.env.example` file to `.env` and adjust values according to your needs:
```bash
cp .env.example .env
```

Then edit `.env` with your configuration:
```env
MODEL_NAME=llama-3.2-3b-instruct-q8_0.gguf
EMBEDDING_MODEL_NAME=nomic-embed-text-v1.5.Q5_K_M.gguf
MODELS_DIR=./models
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

For a complete list of available environment variables, see `.env.example`.

2. **ChromaDB with Docker**:
```bash
docker-compose up -d
```

Or manually:
```bash
docker build -t chroma-db .
docker run -d -p 8000:8000 -v /path/to/data:/data chroma-db
```

Note: The system will automatically fall back to persistent mode if the Docker server is not available.

## 💻 Usage

1. **Start the System**:
```bash
python main.py --model /path/to/model.gguf
```

Or use the default model configured in `.env`:
```bash
python main.py
```

Additional options:
```bash
python main.py --model /path/to/model.gguf --db /path/to/db --force-cpu
```

2. **Available Commands**:
- `/help [command]`: Show help or specific command help
- `/add <file_path> [--tags=tag1,tag2]`: Add a document to the system
- `/search <query> [--filter=value] [--limit=N]`: Search documents
- `/list [--limit=N] [--filter=value]`: List available documents
- `/delete <id>`: Delete a document from the system
- `/settings <option> <value>`: Modify system configuration
- `/history [--limit=N]`: Show conversation history
- `/clear`: Clear conversation history
- `/context [--size=N]`: Show or manage conversation context size
- `/force_exit`: Force system shutdown

3. **Examples**:
```bash
# Add a document
/add document.pdf --tags=manual,tech

# Search for information
/search "What is machine learning?"

# List documents
/list --limit=5

# View conversation history
/history --limit=10

# Clear history
/clear
```

## 🔧 GPU Optimization

The system is optimized for GPU usage:

- Automatic CUDA detection
- Configurable GPU layer loading
- VRAM usage monitoring
- Batch processing for embeddings
- Efficient memory management

## 📦 Compatible Models

- **LLM**: GGUF models (recommended: LLaMA-2, LLaMA-3, Mistral)
- **Embeddings**: 
  - SentenceTransformers (default: all-MiniLM-L6-v2)
  - Compatible GGUF embedding models

## 🔍 RAG Features

- **Chunking**: Intelligent document splitting
- **Overlap**: Configurable overlap between chunks
- **Embeddings**: Efficient GPU-based generation
- **Similarity Search**: Optimized semantic search
- **Context Window**: Dynamic context window
- **Conversation Memory**: Maintains conversation context

## ⚠️ Limitations and Considerations

- Requires sufficient VRAM for model and embeddings
- ChromaDB must be accessible for operations (or will use persistent mode)
- Some formats may require additional dependencies
- Performance depends on available hardware

## 🛠️ Troubleshooting

1. **CUDA Error**:
```bash
# Verify CUDA installation
nvidia-smi

# Reinstall llama-cpp-python with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python
```

2. **Compilation Issues**:
   - Make sure you have Visual Studio Build Tools installed (Windows)
   - Verify that CUDA is in the PATH
   - Check that CUDA toolkit version matches your GPU driver

3. **ChromaDB Error**:
```bash
# Verify Docker status
docker ps
# Restart container
docker restart <container_id>
```

Note: If ChromaDB server is unavailable, the system will automatically use persistent mode.

4. **Model Loading Issues**:
   - Verify the model path in `.env` or command line argument
   - Check that the model file exists and is accessible
   - Ensure sufficient VRAM/RAM for the model size

## 📜 Logging

The system maintains detailed logs in `rag_system.log`:
- Model loading
- RAG operations
- GPU usage
- Errors and warnings

Logs are written in UTF-8 encoding for proper character display.

## 🤝 Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License
