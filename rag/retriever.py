# rag/retriever.py
"""
Implementation of the retrieval component for the RAG system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from config import NUM_RETRIEVAL_RESULTS, SYSTEM_PROMPT
from rag.chroma_db import ChromaVectorStore
from llm.model import GGUFModel

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Class to handle information retrieval in a RAG system."""
    
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm_model: Optional[GGUFModel] = None,
        num_results: int = NUM_RETRIEVAL_RESULTS,
        system_prompt: str = SYSTEM_PROMPT
    ):
        """
        Initialize the retrieval component.
        
        Args:
            vector_store: Vector store for querying
            llm_model: Language model (optional)
            num_results: Number of results to retrieve
            system_prompt: System prompt for the LLM
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.num_results = num_results
        self.system_prompt = system_prompt
    
    def retrieve(
        self, 
        query: str,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            where_filter: Optional filter for search
            
        Returns:
            List of relevant documents with metadata and scores
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Get similar documents
        documents, metadatas, scores = self.vector_store.query(
            query_text=query,
            n_results=self.num_results,
            where=where_filter
        )
        
        if not documents:
            logger.warning("No relevant documents found")
            return []
        
        # Combine results
        results = []
        for doc, meta, score in zip(documents, metadatas, scores):
            results.append({
                "content": doc,
                "metadata": meta,
                "score": score
            })
        
        logger.info(f"Retrieved {len(results)} relevant documents")
        return results
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieval results as context for the LLM.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context text
        """
        if not results:
            return "No relevant information found for this query."
        
        # Sort by relevance score
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Format context
        context_parts = []
        for i, result in enumerate(sorted_results, 1):
            source = result["metadata"].get("source", "Unknown source")
            filename = result["metadata"].get("filename", "")
            chunk_index = result["metadata"].get("chunk_index", "")
            
            # Format source in readable way
            source_info = f"[Document {i}: {filename}"
            if chunk_index != "":
                source_info += f", part {chunk_index + 1}"
            source_info += "]"
            
            # Add chunk with its source
            context_parts.append(f"{source_info}\n{result['content']}\n")
        
        # Combine all chunks
        context = "\n".join(context_parts)
        
        return context
    
    def generate_answer(
        self, 
        query: str, 
        context: str
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved context.
        
        Args:
            query: Original query
            context: Formatted retrieved context
            
        Returns:
            Dictionary with generated answer and metadata
        """
        if not self.llm_model:
            logger.error("No language model provided")
            return {"text": "Error: Language model not available", "context_used": False}
        
        # Create full prompt with context
        full_prompt = f"""
To answer the following question, use only the information provided in the context.
If the necessary information is not in the context, indicate that you don't have enough information.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        
        # Generate answer
        response = self.llm_model.generate(
            prompt=full_prompt,
            system_prompt=self.system_prompt
        )
        
        # Add information about context used
        response["context_used"] = True
        response["context"] = context
        
        return response
    
    def query_with_rag(
        self, 
        query: str,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform a complete RAG query: retrieve + generate answer.
        
        Args:
            query: Query text
            where_filter: Optional filter for search
            
        Returns:
            Dictionary with generated answer and metadata
        """
        # Step 1: Retrieve relevant documents
        results = self.retrieve(query, where_filter)
        
        # Step 2: Format context
        context = self.format_context(results)
        
        # Step 3: Generate answer
        if self.llm_model and results:
            answer = self.generate_answer(query, context)
        elif self.llm_model and not results:
            # If no results but we have model
            answer = self.llm_model.generate(
                prompt=f"I don't have specific information about: {query}. Please answer as best you can.",
                system_prompt=self.system_prompt
            )
            answer["context_used"] = False
        else:
            # No model and no results
            answer = {
                "text": "No relevant information found and no model available.",
                "context_used": False
            }
        
        # Add metadata about retrieved documents
        answer["num_docs_retrieved"] = len(results)
        if results:
            answer["doc_scores"] = [r["score"] for r in results]
            answer["doc_sources"] = [r["metadata"].get("filename", "unknown") for r in results]
        
        return answer
