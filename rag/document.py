"""
Functionality to process and split documents into chunks.
"""

import os
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import re
from pathlib import Path

# Libraries to process different document types
import docx2txt
import PyPDF2
import markdown
from bs4 import BeautifulSoup

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class to handle document processing and splitting."""
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path: Union[str, Path]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process a file and split it into chunks.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Tuple of (chunks, metadatas)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        file_name = file_path.name
        file_size = file_path.stat().st_size
        
        # Extract text according to file type
        try:
            if file_ext == '.txt':
                text = self._extract_text_from_txt(file_path)
            elif file_ext == '.pdf':
                text = self._extract_text_from_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                text = self._extract_text_from_docx(file_path)
            elif file_ext in ['.md', '.markdown']:
                text = self._extract_text_from_markdown(file_path)
            elif file_ext in ['.html', '.htm']:
                text = self._extract_text_from_html(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Split text into chunks
            chunks = self._split_text(text)
            
            # Generate metadata for each chunk
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "source": str(file_path),
                    "filename": file_name,
                    "filetype": file_ext[1:],  # Remove initial dot
                    "filesize": file_size,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            logger.info(f"File {file_name} processed into {len(chunks)} chunks")
            return chunks, metadatas
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            raise
    
    def _extract_text_from_txt(self, file_path: Path) -> str:
        """Extract text from .txt files."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from Word files."""
        return docx2txt.process(file_path)
    
    def _extract_text_from_markdown(self, file_path: Path) -> str:
        """Extract text from Markdown files."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            md_text = f.read()
        
        # Convert markdown to HTML and then to plain text
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML files."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            html = f.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        # Remove scripts, styles, etc.
        for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
            script.extract()
        
        return soup.get_text()
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split a text into chunks with overlap.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is shorter than chunk size, return it directly
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine chunk end
            end = start + self.chunk_size
            
            # Adjust to sentence or paragraph end if possible
            if end < len(text):
                # Find nearest sentence end
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('\n', start, end)
                )
                
                # If we found a sentence end, use it
                if sentence_end > start:
                    end = sentence_end + 1
            
            # Ensure end doesn't exceed text length
            end = min(end, len(text))
            
            # Add chunk
            chunks.append(text[start:end].strip())
            
            # Move start considering overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end - self.chunk_overlap)
        
        return chunks
    
    def generate_document_id(self, source: str, chunk_index: int) -> str:
        """
        Generate a unique ID for a document chunk.
        
        Args:
            source: Document source
            chunk_index: Chunk index
            
        Returns:
            Unique ID
        """
        # Use combination of source and index for better traceability
        base = os.path.basename(source)
        return f"{base}_{chunk_index}_{uuid.uuid4().hex[:8]}"
