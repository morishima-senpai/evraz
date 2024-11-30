import os
import re
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.preprocessing import normalize
import multiprocessing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader



class EmbeddingService:
    """
    Advanced multi-project contextual code embedding generator
    Combines parallel processing, contextual understanding, and semantic analysis
    """
    

    def __init__(
        self,
        model_name: str = 'microsoft/codebert-base-mlm',
        data_folder : str = "./review-data",
        gen_embeedings : bool = False,
        max_length : int = 512,
        chunk_size : int = 512,
        chunk_overlap : int = 128,
        context_window : int =3,
        max_workers : int = None,
        output_folder : str = "./store/"
    ):  
       
        """
        Initialize advanced multi-project code embedder
        
        Args:
            model_name (str): Pretrained transformer model for code
            data_folder (str) : Where all the folder resides
            gen_embeedings (bool) : Generate new embeddings or not (`deafult = False`)
            max_length (int): Maximum sequence length
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            context_window (int): Number of surrounding code blocks to consider
            max_workers (int): Number of parallel processing workers
            output_folder (str) : Where to save the embeddings
        """
        
        # Logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.gen_embeedings = gen_embeedings
        self.projects_folder = data_folder
        self.output_folder = output_folder
        
        # Set number of workers
        self.max_workers = max_workers or (os.cpu_count() or 1)
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
        
        # Configuration parameters
        self.max_length = max_length
        self.context_window = context_window
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', ' ', '']
        )

        if self.gen_embeedings:

            # TODO : generate and save embeddings [Uncomment this Code]
            # embeddings_data = self.generate_project_embeddings(self.projects_folder)
            # self.save_embeddings(embeddings_data, self.output_folder)

            # for fast and to check if everything is working or not
            mew()

    def _preprocess_code(self, code: str) -> str:
        """
        Advanced code preprocessing with semantic preservation
        
        Args:
            code (str): Raw code content
        
        Returns:
            str: Semantically cleaned code
        """
        # Remove comments while preserving docstrings
        # code = re.sub(r'(?<!\"\"\"|\'\'\')#.*$', '', code, flags=re.MULTILINE)
        
        # Normalize whitespaces
        # code = re.sub(r'\s+', ' ', code).strip()
        
        # Remove excessive blank lines
        code = re.sub(r'\n{3,}', '\n\n', code)
        
        return code
    
    def _extract_code_blocks(self, file_contents: str) -> List[str]:
        """
        Extract meaningful code blocks with context
        
        Args:
            file_contents (str): Entire file contents
        
        Returns:
            List of code blocks
        """
        # Split by function and class definitions
        blocks = re.split(r'(def |class )', file_contents)
        blocks = [
            ''.join(blocks[i:i+2]) 
            for i in range(0, len(blocks), 2)
        ]
        
        return [block for block in blocks if block.strip()]
    
    def _process_single_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file and generate contextual embeddings
        
        Args:
            file_path (str): Path to the Python file
        
        Returns:
            Dict with file embeddings and metadata
        """
        try:
            file_contents = ""
            if file_path.endswith(".pdf"):
                reader = PdfReader(file_path)
                file_contents = ''.join(page.extract_text() for page in reader.pages)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_contents = f.read()
            
            # Preprocess code
            preprocessed_code = self._preprocess_code(file_contents)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(preprocessed_code)

            if not code_blocks:
                self.logger.warning(f"No valid code blocks in {file_path}, skipping")
                return None
            
            # Generate block embeddings
            block_embeddings = self._generate_block_embeddings(code_blocks)
            
            # Prepare metadata
            metadata = {
                'file_path': file_path,
                'num_blocks': len(code_blocks),
                'total_lines': len(file_contents.splitlines())
            }

            self.logger.info(f"Successfully processed : {file_path}")
            
            return {
                'embeddings': block_embeddings,
                'metadata': metadata
            }
        
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _generate_block_embeddings(
        self, 
        code_blocks: List[str]
    ) -> np.ndarray:
        """
        Generate contextual embeddings for code blocks
        
        Args:
            code_blocks (List[str]): List of code blocks
        
        Returns:
            np.ndarray: Contextual embeddings
        """
        block_embeddings = []
        
        for i, block in enumerate(code_blocks):
            # Contextual windowing
            context_start = max(0, i - self.context_window)
            context_end = min(len(code_blocks), i + self.context_window + 1)
            context_blocks = code_blocks[context_start:context_end]
            
            # Concatenate blocks with context
            context_code = '\n'.join(context_blocks)
            
            # Tokenize with context
            inputs = self.tokenizer(
                context_code, 
                return_tensors='pt', 
                max_length=self.max_length, 
                truncation=True
            )
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            
            block_embeddings.append(embedding)
        
        # Stack and normalize embeddings
        block_embeddings = np.vstack(block_embeddings)
        block_embeddings = normalize(block_embeddings)
        
        return block_embeddings
    
    def generate_project_embeddings(
        self, 
        data_path: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive embeddings for multiple projects
        
        Args:
            data_path (str): Path to directory containing projects
        
        Returns:
            Dict with project embeddings and metadata
        """
        # Find project directories
        project_paths = [
            os.path.join(data_path, d) 
            for d in os.listdir(data_path) 
            # if os.path.isdir(os.path.join(data_path, d))
        ]
        
        self.logger.info(f"Found {len(project_paths)} projects to process")
        
        # Collect all Python files
        
        python_files = []

        for project_path in project_paths:
            
            if project_path.endswith(".pdf"):
                python_files.append(project_path)
                continue

            if project_path.endswith(".txt"):
                python_files.append(project_path)
                continue

            for root, _, files in os.walk(project_path):
                for file in files:
                    python_files.append(os.path.join(root, file))
        
        # Parallel processing of files
        with multiprocessing.Pool(self.max_workers) as pool:
            file_results = pool.map(self._process_single_file, python_files)
        
        # Filter out None results
        file_results = [r for r in file_results if r is not None]

        if not file_results:
            self.logger.error("No embeddings could be generated")
            return {
                'faiss_index': None,
                'metadata': [],
                'embeddings': np.array([])
            }
        
        # Combine embeddings and metadata
        all_embeddings = np.vstack([r['embeddings'] for r in file_results])
        all_metadata = [r['metadata'] for r in file_results]
        
        # Create FAISS index
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(all_embeddings)
        
        return {
            'faiss_index': index,
            'metadata': all_metadata,
            'embeddings': all_embeddings
        }
    
    def save_embeddings(
        self, 
        embeddings_data: Dict[str, Any], 
        output_path: str
    ):
        """
        Save generated embeddings
        
        Args:
            embeddings_data (Dict): Embedding data
            output_path (str): Path to save embeddings
        """
        os.makedirs(output_path, exist_ok=True)


        unique_metadata = {meta['file_path']: meta for meta in embeddings_data['metadata']}.values()
        all_metadata = list(unique_metadata)
        
        # Save FAISS index
        faiss.write_index(
            embeddings_data['faiss_index'], 
            os.path.join(output_path, 'index.faiss')
        )
        
        # Save additional metadata
        import json, pickle
        with open(os.path.join(output_path, 'index.json'), 'w') as f:
            json.dump( all_metadata , f) # embeddings_data['metadata'], f)

        with open(os.path.join(output_path, 'index.pkl'), 'wb') as f:
           pickle.dump({
                'faiss_index': embeddings_data['faiss_index'],
                'metadata': embeddings_data['metadata'] ,# all_metadata,
                'embeddings': embeddings_data['embeddings']
            }, f)

        
        # Save raw embeddings for potential future use
        np.save(
            os.path.join(output_path, 'raw_embeddings.npy'), 
            embeddings_data['embeddings']
        )


def mew():
    x = HuggingFaceEmbeddings(model_name="microsoft/codebert-base-mlm")
    reader = PyPDFLoader("./boo/Python.pdf")
    k = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128, separators=['\n\n', '\n', ' ', ''])
    # file_contents = ''.join(page.extract_text() for page in reader.pages)
    documents = k.split_documents(reader.load())
    y = FAISS.from_documents(documents, x)
    y.save_local("./gen/")
