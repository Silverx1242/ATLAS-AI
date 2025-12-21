# commands/executor.py
"""
Implementation of the command executor for the system.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from commands.parser import CommandParser
from rag.document import DocumentProcessor
from rag.chroma_db import ChromaVectorStore
from rag.retriever import RAGRetriever
from llm.model import GGUFModel
from config import TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS

logger = logging.getLogger(__name__)

class CommandExecutor:
    """Class to execute system commands."""
    
    def __init__(
        self,
        command_parser: CommandParser,
        document_processor: DocumentProcessor,
        vector_store: ChromaVectorStore,
        rag_retriever: RAGRetriever,
        llm_model: Optional[GGUFModel] = None,
        rag_system: Optional[Any] = None  # Add rag_system parameter
    ):
        """
        Initialize the command executor.
        
        Args:
            command_parser: Command parser
            document_processor: Document processor
            vector_store: Vector store
            rag_retriever: Retrieval component
            llm_model: Language model (optional)
            rag_system: Main RAG system (optional)
        """
        self.parser = command_parser
        self.doc_processor = document_processor
        self.vector_store = vector_store
        self.rag_retriever = rag_retriever
        self.llm_model = llm_model
        self.rag_system = rag_system  # Store rag_system reference
        
        # Map commands to execution functions
        self.command_handlers = {
            "help": self._execute_help,
            "search": self._execute_search,
            "add": self._execute_add,
            "delete": self._execute_delete,
            "list": self._execute_list,
            "settings": self._execute_settings,
            "history": self._execute_history,
            "clear": self._execute_clear,
            "context": self._execute_context,
            "force_exit": self._execute_force_exit
        }
    
    def execute_command(self, command_text: str) -> Dict[str, Any]:
        """
        Execute a command.
        
        Args:
            command_text: Command text
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Check if it's a command
            if not self.parser.is_command(command_text):
                return {
                    "success": False,
                    "message": f"Not a valid command: {command_text}",
                    "is_command": False
                }
            
            # Parse the command
            command_name, args, kwargs = self.parser.parse_command(command_text)
            
            # Check if command exists
            if command_name not in self.command_handlers:
                return {
                    "success": False,
                    "message": f"Command not recognized: {command_name}",
                    "is_command": True
                }
            
            # Execute the command
            handler = self.command_handlers[command_name]
            result = handler(args, kwargs)
            
            # Ensure result includes required fields
            if "success" not in result:
                result["success"] = True
            result["is_command"] = True
            result["command_name"] = command_name
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return {
                "success": False,
                "message": f"Error executing command: {str(e)}",
                "is_command": True,
                "error": str(e)
            }
    
    def _execute_help(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the help command."""
        specific_command = args[0] if args else None
        help_text = self.parser.get_help_text(specific_command)
        
        return {
            "message": help_text,
            "command": specific_command
        }
    
    def _execute_search(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the search command."""
        if not args:
            return {
                "success": False,
                "message": "A search term is required"
            }
        
        query = " ".join(args)
        limit = int(kwargs.get("limit", 5))
        
        # Process filters if they exist
        where_filter = None
        if "filter" in kwargs:
            filter_str = kwargs["filter"]
            if "=" in filter_str:
                key, value = filter_str.split("=", 1)
                where_filter = {key.strip(): value.strip()}
        
        # Perform search
        results = self.rag_retriever.retrieve(query, where_filter)
        
        # Limit results
        results = results[:limit]
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "message": f"Found {len(results)} relevant documents"
        }
    
    def _execute_add(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the command to add documents."""
        if not args:
            return {
                "success": False,
                "message": "File path is required"
            }
        
        file_path = args[0]
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"File does not exist: {file_path}"
            }
        
        # Process tags if they exist
        tags = []
        if "tags" in kwargs:
            tags = [tag.strip() for tag in kwargs["tags"].split(",")]
        
        try:
            # Process the document
            chunks, metadatas = self.doc_processor.process_file(file_path)
            
            # Add tags if provided
            if tags:
                for metadata in metadatas:
                    metadata["tags"] = tags
            
            # Generate IDs for chunks
            ids = []
            for i, _ in enumerate(chunks):
                chunk_id = self.doc_processor.generate_document_id(file_path, i)
                ids.append(chunk_id)
            
            # Store in vector database
            self.vector_store.add_documents(chunks, ids, metadatas)
            
            return {
                "success": True,
                "message": f"Document added: {file_path}",
                "chunks": len(chunks),
                "file_path": file_path,
                "ids": ids
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return {
                "success": False,
                "message": f"Error adding document: {str(e)}",
                "file_path": file_path
            }
    
    def _execute_delete(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the command to delete documents."""
        if not args:
            return {
                "success": False,
                "message": "Document ID is required"
            }
        
        doc_id = args[0]
        
        try:
            # Delete document
            self.vector_store.delete([doc_id])
            
            return {
                "success": True,
                "message": f"Document deleted: {doc_id}",
                "id": doc_id
            }
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return {
                "success": False,
                "message": f"Error deleting document: {str(e)}",
                "id": doc_id
            }
    
    def _execute_list(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the command to list documents."""
        limit = int(kwargs.get("limit", 10))
        
        # Process filters if they exist
        where_filter = None
        if "filter" in kwargs:
            filter_str = kwargs["filter"]
            if "=" in filter_str:
                key, value = filter_str.split("=", 1)
                where_filter = {key.strip(): value.strip()}
        
        # Get documents
        results = self.vector_store.get_all_documents(limit, where_filter)
        
        return {
            "success": True,
            "message": f"Found {len(results['ids'])} documents",
            "documents": results,
            "count": len(results['ids'])
        }
    
    def _execute_settings(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Execute the command to modify configuration."""
        if len(args) < 2:
            return {
                "success": False,
                "message": "Option and value are required"
            }
        
        option = args[0].lower()
        value = args[1]
        
        # Check if model is available
        if not self.llm_model:
            return {
                "success": False,
                "message": "No model available to modify configuration"
            }
        
        valid_options = {
            "temperature": (float, 0.0, 2.0, TEMPERATURE),
            "top_p": (float, 0.0, 1.0, TOP_P),
            "top_k": (int, 1, 100, TOP_K),
            "max_tokens": (int, 1, 4096, MAX_TOKENS)
        }
        
        # Check if option is valid
        if option not in valid_options:
            valid_opts = ", ".join(valid_options.keys())
            return {
                "success": False,
                "message": f"Invalid option. Available options: {valid_opts}"
            }
        
        # Get type and ranges
        val_type, min_val, max_val, default_val = valid_options[option]
        
        try:
            # Convert value to correct type
            typed_value = val_type(value)
            
            # Check range
            if typed_value < min_val or typed_value > max_val:
                return {
                    "success": False,
                    "message": f"Value out of range. Valid range: {min_val} - {max_val}"
                }
            
            # Note: Dynamic configuration changes are not fully implemented.
            # For persistent changes, modify the .env file and restart the system.
            # Current values are read from config.py which loads from .env at startup.
            
            return {
                "success": True,
                "message": f"Configuration validated: {option} = {typed_value} (previous value: {default_val})\n"
                          f"NOTE: To apply persistent changes, modify the {option.upper()} variable in the .env file and restart the system.\n"
                          f"Dynamic changes during the session are not currently implemented.",
                "option": option,
                "value": typed_value,
                "previous_value": default_val
            }
            
        except ValueError:
            return {
                "success": False,
                "message": f"Invalid value for {option}. Expected {val_type.__name__}"
            }
    
    def _execute_history(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Show conversation history."""
        try:
            if not self.rag_system or not hasattr(self.rag_system, 'conversation_history'):
                return {
                    "success": True,
                    "message": "No history available.",
                    "is_command": True
                }

            history = self.rag_system.conversation_history
            if not history:
                return {
                    "success": True,
                    "message": "History is empty.",
                    "is_command": True
                }

            # Process limit if it exists
            limit = None
            if "limit" in kwargs:
                try:
                    limit = int(kwargs["limit"])
                    history = history[-limit:]
                except ValueError:
                    return {
                        "success": False,
                        "message": "Limit must be a number",
                        "is_command": True
                    }

            # Format history
            formatted_history = "\nConversation history:\n" + "="*50 + "\n"
            for msg in history:
                timestamp = datetime.fromtimestamp(msg['timestamp']).strftime('%H:%M:%S')
                formatted_history += f"[{timestamp}] {msg['role'].upper()}\n"
                formatted_history += f"{msg['content']}\n"
                formatted_history += "-"*50 + "\n"

            return {
                "success": True,
                "message": formatted_history,
                "is_command": True
            }

        except Exception as e:
            logger.error(f"Error in history command: {str(e)}")
            return {
                "success": False,
                "message": f"Error showing history: {str(e)}",
                "is_command": True
            }
    
    def _execute_clear(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Clear conversation history."""
        try:
            if not self.rag_system or not hasattr(self.rag_system, 'conversation_history'):
                return {
                    "success": False,
                    "message": "No history available to clear.",
                    "is_command": True
                }

            history_length = len(self.rag_system.conversation_history)
            self.rag_system.conversation_history = []
            
            return {
                "success": True,
                "message": f"History cleared. Removed {history_length} messages.",
                "is_command": True
            }

        except Exception as e:
            logger.error(f"Error in clear command: {str(e)}")
            return {
                "success": False,
                "message": f"Error clearing history: {str(e)}",
                "is_command": True
            }
    
    def _execute_context(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Show or modify conversation context size."""
        try:
            from config import CONTEXT_WINDOW_MESSAGES, MAX_HISTORY_LENGTH
            
            # If --size is provided, modify context size
            if "size" in kwargs:
                try:
                    new_size = int(kwargs["size"])
                    if new_size < 1:
                        return {
                            "success": False,
                            "message": "Context size must be at least 1.",
                            "is_command": True
                        }
                    if new_size > MAX_HISTORY_LENGTH:
                        return {
                            "success": False,
                            "message": f"Context size cannot exceed {MAX_HISTORY_LENGTH} (MAX_HISTORY_LENGTH).",
                            "is_command": True
                        }
                    
                    # Note: This only changes temporarily, doesn't persist
                    # To change permanently, would need to modify config.py or use environment variables
                    # For now, we just inform that the environment variable needs to be changed
                    return {
                        "success": True,
                        "message": f"To change context size permanently, set the environment variable CONTEXT_WINDOW_MESSAGES={new_size}. Current value: {CONTEXT_WINDOW_MESSAGES}",
                        "is_command": True,
                        "current_size": CONTEXT_WINDOW_MESSAGES,
                        "requested_size": new_size
                    }
                except ValueError:
                    return {
                        "success": False,
                        "message": "Size must be an integer.",
                        "is_command": True
                    }
            
            # If --size is not provided, show current context information
            if not self.rag_system or not hasattr(self.rag_system, 'conversation_history'):
                return {
                    "success": True,
                    "message": f"Context size: {CONTEXT_WINDOW_MESSAGES} messages\nMaximum history: {MAX_HISTORY_LENGTH} messages\nNo history available.",
                    "is_command": True
                }
            
            history_length = len(self.rag_system.conversation_history)
            context_info = f"Context size: {CONTEXT_WINDOW_MESSAGES} messages\n"
            context_info += f"Maximum history: {MAX_HISTORY_LENGTH} messages\n"
            context_info += f"Messages in current history: {history_length}\n"
            context_info += f"Messages used in context: {min(history_length, CONTEXT_WINDOW_MESSAGES)}"
            
            return {
                "success": True,
                "message": context_info,
                "is_command": True,
                "context_size": CONTEXT_WINDOW_MESSAGES,
                "history_length": history_length
            }

        except Exception as e:
            logger.error(f"Error in context command: {str(e)}")
            return {
                "success": False,
                "message": f"Error managing context: {str(e)}",
                "is_command": True
            }
    
    def _execute_force_exit(self, args: List[str], kwargs: Dict[str, str]) -> Dict[str, Any]:
        """Force system shutdown."""
        logger.info("force_exit command executed")
        return {
            "success": True,
            "message": "Forcing system shutdown...",
            "is_command": True,
            "force_exit": True  # Special flag to indicate forced exit
        }
