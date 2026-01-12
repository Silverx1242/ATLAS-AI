from typing import Dict, List, Optional, Union
import logging
import os
import sys
import time
from llama_cpp import Llama
import torch
import pynvml  # Add NVML for GPU monitoring

from config import (
    MODEL_PATH, MODEL_CONTEXT_SIZE, MODEL_N_THREADS, MODEL_N_GPU_LAYERS,
    TEMPERATURE, TOP_P, TOP_K, MAX_TOKENS, REPEAT_PENALTY, GPU_MAIN_DEVICE,
    N_BATCH, GPU_MEMORY_UTILIZATION
)

logger = logging.getLogger(__name__)

class GGUFModel:
    """Class to handle interaction with GGUF models using GPU."""
    
    def __init__(
        self,
        model_path: Optional[str] = MODEL_PATH,
        n_ctx: int = MODEL_CONTEXT_SIZE,
        n_threads: int = MODEL_N_THREADS,
        n_gpu_layers: int = MODEL_N_GPU_LAYERS,
        main_gpu: int = GPU_MAIN_DEVICE,
        tensor_split: Optional[List[float]] = None
    ):
        """Initialize the GGUF model."""
        if model_path is None:
            raise ValueError("A model path (model_path) is required")
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = main_gpu
        self.tensor_split = tensor_split
        
        # Initialize NVML
        self.gpu_name = None
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.cuda_available = True
            gpu_name_raw = pynvml.nvmlDeviceGetName(self.handle)
            # Handle both bytes and strings (depending on pynvml version)
            if isinstance(gpu_name_raw, bytes):
                self.gpu_name = gpu_name_raw.decode('utf-8')
            else:
                self.gpu_name = str(gpu_name_raw)
            logger.info(f"GPU detected: {self.gpu_name}")
        except Exception as e:
            logger.warning(f"Could not initialize NVML: {e}")
            # Try to check CUDA with torch only if available, but don't depend on it
            try:
                self.cuda_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            except:
                self.cuda_available = False
            self.handle = None
        
        self._load_model()

    def _get_gpu_memory(self):
        """Get current GPU memory usage."""
        try:
            if self.handle:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return {
                    'total': info.total / (1024**3),  # Convert to GB
                    'used': info.used / (1024**3),
                    'free': info.free / (1024**3)
                }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
        return None

    def _load_model(self) -> None:
        """Load the model with GGML CUDA optimizations."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Get initial GPU memory state
            initial_mem = self._get_gpu_memory()
            if initial_mem:
                logger.info(f"Initial GPU memory: {initial_mem['used']:.2f}GB / {initial_mem['total']:.2f}GB")
            
            # Configure model parameters
            if self.cuda_available:
                # Use GPU name from NVML if available
                gpu_info = f"GPU detected: {self.gpu_name}" if self.gpu_name else "GPU detected (CUDA available)"
                logger.info(gpu_info)
                
                # Determine number of GPU layers
                if self.n_gpu_layers < 0:
                    # -1 means use all layers on GPU
                    n_gpu_layers = -1
                    logger.info("Configuring all layers for GPU (n_gpu_layers=-1)")
                else:
                    n_gpu_layers = self.n_gpu_layers
                    logger.info(f"Configuring {n_gpu_layers} layers for GPU")
                
                model_params = {
                    "model_path": self.model_path,
                    "n_ctx": self.n_ctx,
                    "n_threads": self.n_threads,
                    "n_gpu_layers": n_gpu_layers,
                    "n_batch": N_BATCH,
                    "f16_kv": True,
                    "use_mmap": False,
                    "use_mlock": False,
                    "vocab_only": False,
                    "seed": -1,
                    "verbose": True
                }
                
                # Remove unsupported parameters
                if hasattr(self, 'tensor_split'): delattr(self, 'tensor_split')
                if hasattr(self, 'main_gpu'): delattr(self, 'main_gpu')
                
            else:
                model_params = {
                    "model_path": self.model_path,
                    "n_ctx": self.n_ctx,
                    "n_threads": self.n_threads,
                    "n_gpu_layers": 0,
                    "use_mmap": True,
                    "use_mlock": False
                }

            # Load model
            logger.info("Starting model load into memory...")
            # Determine number of GPU layers for logging
            actual_n_gpu_layers = self.n_gpu_layers if self.n_gpu_layers >= 0 else -1
            if self.cuda_available and actual_n_gpu_layers != 0:
                logger.info(f"Attempting to load model with {actual_n_gpu_layers} layers on GPU...")
            self.model = Llama(**model_params)
            
            # Wait for model to load
            time.sleep(2)
            
            # Determine if model is using GPU
            # Check if it actually loaded on GPU by checking memory
            if self.cuda_available and actual_n_gpu_layers != 0:
                # Check GPU memory to confirm it was used
                final_mem = self._get_gpu_memory()
                if final_mem and initial_mem:
                    mem_diff = final_mem['used'] - initial_mem['used']
                    # If there's a significant memory increase (>100MB), assume it's on GPU
                    if mem_diff > 0.1:  # More than 100MB
                        self.using_gpu = True
                        logger.info(f"GPU memory after load: {final_mem['used']:.2f}GB")
                        logger.info(f"Memory increase: {mem_diff:.2f}GB")
                        device_msg = f"GPU [OK] (VRAM: {mem_diff:.2f}GB)"
                    else:
                        self.using_gpu = False
                        logger.warning(f"GPU detected but model loaded on CPU. VRAM increase: {mem_diff:.2f}GB")
                        logger.warning("This may indicate that llama-cpp-python is not compiled with CUDA support.")
                        logger.warning("To use GPU, reinstall llama-cpp-python with: pip install llama-cpp-python --force-reinstall --no-cache-dir")
                        device_msg = "CPU (GPU not available in llama-cpp-python)"
                else:
                    # If we can't verify memory, assume GPU if configured
                    self.using_gpu = True
                    device_msg = "GPU [OK]"
                logger.info(f"Model loaded on {device_msg}")
            else:
                self.using_gpu = False
                logger.info("Model loaded on CPU")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        max_tokens: int = MAX_TOKENS,
        repeat_penalty: float = REPEAT_PENALTY,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Union[str, int]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            system_prompt: System message (if model supports it)
            temperature: Temperature for generation
            top_p: Value for nucleus sampling
            top_k: Number of tokens to consider
            max_tokens: Maximum number of tokens to generate
            repeat_penalty: Repetition penalty
            stop: List of sequences to stop generation
            
        Returns:
            Dictionary with generated text and metadata
        """
        if not self.model:
            raise ValueError("Model is not loaded")
        
        if stop is None:
            stop = []
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.debug(f"Generating response for prompt: {prompt[:50]}...")
            
            # Monitor VRAM usage before inference
            if torch.cuda.is_available() and self.using_gpu:
                mem_before = torch.cuda.memory_allocated(0) / (1024**3)
                logger.debug(f"VRAM before inference: {mem_before:.2f} GB")
            
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
                stop=stop
            )
            
            # Monitor VRAM usage after inference
            if torch.cuda.is_available() and self.using_gpu:
                mem_after = torch.cuda.memory_allocated(0) / (1024**3)
                logger.debug(f"VRAM after inference: {mem_after:.2f} GB")
            
            generated_text = response["choices"][0]["message"]["content"]
            
            return {
                "text": generated_text,
                "tokens_used": response["usage"]["total_tokens"],
                "finish_reason": response["choices"][0]["finish_reason"]
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return {
                "text": f"Error generating response: {str(e)}",
                "tokens_used": 0,
                "finish_reason": "error"
            }
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embeddings from text using the model.
        
        Args:
            text: Text to get embeddings from
            
        Returns:
            List of float values representing the embedding
        """
        if not self.model:
            raise ValueError("Model is not loaded")
        
        try:
            embedding = self.model.embed(text)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise