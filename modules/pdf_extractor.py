"""
PDF text extraction module
"""
import os
from typing import List, Dict
from pathlib import Path
import PyPDF2
from utils.logger import log
from config.settings import settings


class PDFExtractor:
    """Extract text content from PDF files"""
    
    def __init__(self):
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Validate file
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            file_size = os.path.getsize(pdf_path)
            if file_size > self.max_file_size:
                raise ValueError(f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB")
            
            log.info(f"Extracting text from: {pdf_path}")
            
            # Extract text
            text_content = []
            metadata = {
                "source": pdf_path,
                "filename": Path(pdf_path).name,
                "num_pages": 0,
                "file_size_bytes": file_size
            }
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata["num_pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append({
                                "page": page_num,
                                "content": text.strip()
                            })
                    except Exception as e:
                        log.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        continue
            
            full_text = "\n\n".join([page["content"] for page in text_content])
            
            log.info(f"Successfully extracted {len(full_text)} characters from {metadata['num_pages']} pages")
            
            return {
                "text": full_text,
                "pages": text_content,
                "metadata": metadata
            }
            
        except Exception as e:
            log.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_from_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, any]]:
        """
        Extract text from multiple PDF files
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        results = []
        
        for pdf_path in pdf_paths:
            try:
                result = self.extract_text_from_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                log.error(f"Failed to process {pdf_path}: {str(e)}")
                continue
        
        log.info(f"Processed {len(results)}/{len(pdf_paths)} PDF files successfully")
        return results
    
    def extract_from_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Extract text from all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        pdf_files = list(Path(directory_path).glob("**/*.pdf"))
        log.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        return self.extract_from_multiple_pdfs([str(f) for f in pdf_files])
