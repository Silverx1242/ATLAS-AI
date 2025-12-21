# rag/chroma_db.py
"""
Implementation of ChromaDB integration for vector storage
connecting to a ChromaDB instance in Docker or using persistent mode as fallback.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings

from config import CHROMA_COLLECTION_NAME, SIMILARITY_THRESHOLD, CHROMA_HOST, CHROMA_PORT, CHROMA_DB_PATH

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Class to handle vector storage with ChromaDB."""
    
    def __init__(
        self, 
        host: str = CHROMA_HOST,
        port: int = CHROMA_PORT,
        collection_name: str = CHROMA_COLLECTION_NAME,
        db_path: Optional[str] = None
    ):
        """
        Initialize connection with ChromaDB.
        Tries to connect to HTTP server first, if it fails uses persistent mode.
        
        Args:
            host: ChromaDB server host (only for server mode)
            port: ChromaDB server port (only for server mode)
            collection_name: Collection name
            db_path: Path for persistent storage (fallback)
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.db_path = db_path or CHROMA_DB_PATH
        self.use_persistent = False
        
        # Use the same embedding model as AnythingLLM
        from chromadb.utils import embedding_functions
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Try to connect to HTTP server first
        try:
            logger.info(f"Attempting to connect to ChromaDB at {host}:{port}...")
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            # Test connection
            self.client.heartbeat()
            logger.info(f"Connected to ChromaDB server at {host}:{port}")
            
            # Create or retrieve collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.ef
            )
            
            logger.info(f"Collection '{self.collection_name}' initialized in server mode")
            
        except Exception as e:
            logger.warning(f"Could not connect to ChromaDB server: {str(e)}")
            logger.info(f"Using persistent mode at: {self.db_path}")
            
            # Fallback to persistent mode
            try:
                # Create directory if it doesn't exist
                Path(self.db_path).mkdir(parents=True, exist_ok=True)
                
                self.client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
                self.use_persistent = True
                
                # Create or retrieve collection
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.ef
                )
                
                logger.info(f"Collection '{self.collection_name}' initialized in persistent mode")
                
            except Exception as persistent_error:
                logger.error(f"Error initializing ChromaDB in persistent mode: {str(persistent_error)}")
                raise RuntimeError(f"Could not initialize ChromaDB in either server or persistent mode. Persistent error: {str(persistent_error)}")
    
    def add_documents(
        self, 
        texts: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the vector database.
        
        Args:
            texts: List of texts to add
            ids: List of unique IDs for each text
            metadatas: List of metadata for each text (optional)
        """
        if not texts or not ids:
            logger.warning("No texts or IDs provided to add")
            return
        
        if len(texts) != len(ids):
            raise ValueError("The number of texts and IDs must be equal")
        
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("The number of metadatas must equal the number of texts")
        
        if not metadatas:
            metadatas = [{} for _ in texts]
        
        try:
            self.collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"Added {len(texts)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(
        self, 
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Query documents similar to a query text.
        """
        try:
            # Perform query directly without processing embeddings
            query_params = {
                "query_texts": [query_text],
                "n_results": n_results,
                "where": where,
                "include": ['documents', 'metadatas', 'distances']
            }

            # If we're using a custom embedding function
            if hasattr(self.ef, 'model'):
                # Generate embedding manually
                query_embedding = self.ef([query_text])[0]
                query_params["query_embeddings"] = [query_embedding]
                del query_params["query_texts"]  # Remove text if we use direct embedding

            results = self.collection.query(**query_params)
            
            # Check if there are results
            if not results or 'documents' not in results or not results['documents']:
                logger.info("No results found")
                return [], [], []
            
            # Extract result components
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = [float(d) for d in results['distances'][0]]
            
            # Convert distances to similarities and filter
            filtered_results = [
                (doc, meta, 1.0 - dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if (1.0 - dist) >= threshold
            ]
            
            if not filtered_results:
                logger.info(f"No results found with similarity >= {threshold}")
                return [], [], []
            
            # Unpack results
            docs, metas, sims = zip(*filtered_results)
            
            return list(docs), list(metas), list(sims)
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}", exc_info=True)
            return [], [], []
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents by ID.
        
        Args:
            ids: List of IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def get_all_documents(
        self, 
        limit: int = 100,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """
        Get all stored documents.
        
        Args:
            limit: Maximum number of documents to retrieve
            where: Metadata filter
            
        Returns:
            Dictionary with lists of IDs, documents and metadatas
        """
        try:
            results = self.collection.get(
                limit=limit,
                where=where
            )
            
            return {
                "ids": results.get("ids", []),
                "documents": results.get("documents", []),
                "metadatas": results.get("metadatas", [])
            }
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return {"ids": [], "documents": [], "metadatas": []}
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Log search results
            logger.debug(f"Query: {query}")
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                logger.debug(f"Document {i+1} - Similarity: {1-distance:.4f}")
                logger.debug(f"Content: {doc[:100]}...")
            
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []