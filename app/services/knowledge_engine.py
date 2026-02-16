"""
Knowledge Engine - Semantic search and content retrieval with multilingual support
Primary source: knowledge_base/sample_faqs.json
Fallback: medgemma model for medical questions
"""
from typing import List, Dict, Optional, Tuple, Union, Any
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
import hashlib
import re
import threading
import base64
import io
from pathlib import Path
from sqlalchemy.orm import Session
from app.database.models import FAQ, Article
from app.config import settings
from app.utils.translator import translation_service
import logging

logger = logging.getLogger(__name__)

# Optional PIL for multimodal image input (MedGemma vision)
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PILImage = None


def _decode_image_for_medgemma(image_input: Union[str, bytes, Any]) -> Optional[Any]:
    """
    Decode image input to PIL Image for MedGemma multimodal input.
    Accepts: base64 string, bytes, or PIL Image. Returns PIL Image or None if unavailable/invalid.
    """
    if not PIL_AVAILABLE or PILImage is None:
        logger.warning("PIL not available; multimodal image input disabled")
        return None
    if image_input is None:
        return None
    try:
        if hasattr(image_input, "convert") and callable(getattr(image_input, "convert", None)):
            return image_input.convert("RGB") if image_input.mode != "RGB" else image_input
        if isinstance(image_input, bytes):
            return PILImage.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, str):
            raw = base64.b64decode(image_input, validate=True)
            return PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        logger.warning(f"Image decode failed for MedGemma: {e}")
    return None

# Try to import Redis for caching
try:
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis unavailable - caching disabled")


class MedgemmaModelManager:
    """Thread-safe singleton manager for Medgemma model to prevent multiple loads and memory leaks"""
    
    _instance = None
    _lock = threading.Lock()
    _model_lock = threading.Lock()  # Lock for model operations
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.medgemma_model = None
        self.medgemma_processor = None
        self.medgemma_tokenizer = None
        self.medgemma_loaded = False
        self._reference_count = 0
        self._reference_lock = threading.Lock()
    
    def get_model(self):
        """Get or load the Medgemma model (thread-safe)"""
        with self._model_lock:
            if not self.medgemma_loaded:
                if not settings.USE_MEDGEMMA:
                    logger.info("Medgemma disabled")
                    return None, None, None
                
                logger.info("Loading Medgemma model (shared instance)...")
                self._load_model()
            
            # Increment reference count
            with self._reference_lock:
                self._reference_count += 1
                logger.debug(f"Model reference count: {self._reference_count}")
            
            return self.medgemma_model, self.medgemma_processor, self.medgemma_tokenizer
    
    def release_model(self):
        """Release model reference (decrement counter, unload if no references)"""
        with self._reference_lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                logger.debug(f"Model reference count: {self._reference_count}")
            
            # Don't unload immediately - keep model in memory for reuse
            # Only unload if explicitly requested via cleanup
            return self._reference_count
    
    def _load_model(self):
        """Load the Medgemma model"""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            from pathlib import Path
            
            # Check if local model exists
            local_model_path = Path(settings.MEDGEMMA_MODEL_PATH)
            model_source = None
            
            if settings.USE_LOCAL_MEDGEMMA and local_model_path.exists() and local_model_path.is_dir():
                model_source = str(local_model_path)
                logger.info(f"Loading Medgemma from local path")
            else:
                model_source = settings.MEDGEMMA_MODEL_NAME
                logger.info(f"Loading Medgemma from HuggingFace")
                if settings.USE_LOCAL_MEDGEMMA:
                    logger.warning(f"Local model not found, using HuggingFace")
            
            # Check if accelerate is available for device_map
            try:
                import accelerate
                use_device_map = "auto" if torch.cuda.is_available() else None
            except ImportError:
                logger.warning("accelerate not installed - using default device mapping")
                use_device_map = None
            
            # Load processor
            self.medgemma_processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True
            )
            self.medgemma_tokenizer = self.medgemma_processor.tokenizer
            
            # Load model
            load_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            }
            
            if use_device_map:
                load_kwargs['device_map'] = use_device_map
            
            self.medgemma_model = AutoModelForImageTextToText.from_pretrained(
                model_source,
                **load_kwargs
            )
            
            if not torch.cuda.is_available() and not use_device_map:
                self.medgemma_model = self.medgemma_model.to("cpu")
            
            self.medgemma_model.eval()
            device = next(self.medgemma_model.parameters()).device
            logger.info(f"Medgemma loaded on {device}")
            self.medgemma_loaded = True
        except Exception as e:
            logger.warning(f"Medgemma load failed: {e} - using FAQ-only mode")
            self.medgemma_model = None
            self.medgemma_processor = None
            self.medgemma_tokenizer = None
            self.medgemma_loaded = False
    
    def cleanup(self, force: bool = False):
        """Cleanup model and release GPU memory"""
        with self._model_lock:
            if not self.medgemma_model:
                logger.debug("No model to cleanup")
                return
            
            # Only cleanup if no references or forced
            with self._reference_lock:
                if not force and self._reference_count > 0:
                    logger.debug(f"Skipping cleanup - {self._reference_count} references still active")
                    return
            
            logger.info("Cleaning up Medgemma model and releasing GPU memory...")
            try:
                self._release_gpu_memory()
                
                # Verify cleanup succeeded
                if self.medgemma_model is not None:
                    logger.warning("Model still exists after cleanup, forcing deletion...")
                    import torch
                    import gc
                    try:
                        if hasattr(self.medgemma_model, 'to'):
                            self.medgemma_model = self.medgemma_model.to('cpu')
                        del self.medgemma_model
                        self.medgemma_model = None
                    except Exception as e:
                        logger.error(f"Failed to force delete model: {e}")
                    
                    if self.medgemma_processor:
                        del self.medgemma_processor
                        self.medgemma_processor = None
                    
                    if self.medgemma_tokenizer:
                        del self.medgemma_tokenizer
                        self.medgemma_tokenizer = None
                    
                    for _ in range(5):
                        gc.collect()
                    
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            with torch.cuda.device(i):
                                for _ in range(5):
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()
                                torch.cuda.synchronize()
                    
                    self.medgemma_loaded = False
                    with self._reference_lock:
                        self._reference_count = 0
                    logger.info("Model forcefully deleted after cleanup verification")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)
                # Still try to clear state
                self.medgemma_model = None
                self.medgemma_processor = None
                self.medgemma_tokenizer = None
                self.medgemma_loaded = False
                with self._reference_lock:
                    self._reference_count = 0
    
    def _release_gpu_memory(self):
        """Release GPU memory by deleting model and clearing CUDA cache - AGGRESSIVE"""
        try:
            import torch
            import gc
            
            if not self.medgemma_model:
                return
            
            logger.info("Starting aggressive GPU memory cleanup...")
            
            # Check if model has device_map
            has_device_map = hasattr(self.medgemma_model, 'hf_device_map') or hasattr(self.medgemma_model, 'device_map')
            
            if has_device_map:
                logger.info("Releasing GPU memory for device_map model - using aggressive cleanup")
                try:
                    # Get device map
                    if hasattr(self.medgemma_model, 'hf_device_map'):
                        device_map = self.medgemma_model.hf_device_map
                    elif hasattr(self.medgemma_model, 'device_map'):
                        device_map = self.medgemma_model.device_map
                    else:
                        device_map = None
                    
                    # For device_map models, we need special handling
                    # device_map models can't use .to("cpu") - need to manually move all modules
                    if device_map:
                        logger.info("Moving device_map model to CPU (module-by-module)...")
                        try:
                            # Method 1: Try using accelerate's utilities to properly unload device_map model
                            try:
                                from accelerate import dispatch_model, cpu_offload
                                from accelerate.utils import get_balanced_memory
                                
                                logger.debug("Using accelerate utilities to unload device_map model...")
                                
                                # Try to dispatch model to CPU only
                                try:
                                    # Get model's max memory to determine CPU offload
                                    max_memory = get_balanced_memory(
                                        self.medgemma_model,
                                        max_memory={"cpu": "99GiB"}  # Force everything to CPU
                                    )
                                    # Dispatch model to CPU
                                    self.medgemma_model = dispatch_model(
                                        self.medgemma_model,
                                        device_map={"": "cpu"},
                                        max_memory=max_memory
                                    )
                                    logger.debug("Model dispatched to CPU using accelerate")
                                except Exception as e1:
                                    logger.debug(f"dispatch_model failed: {e1}, trying cpu_offload")
                                    # Fallback to cpu_offload
                                    try:
                                        cpu_offload(self.medgemma_model, execution_device="cpu")
                                        logger.debug("Model offloaded to CPU using accelerate")
                                    except Exception as e2:
                                        logger.debug(f"cpu_offload also failed: {e2}, using manual method")
                                        raise e2
                                
                                gc.collect()
                                if torch.cuda.is_available():
                                    for i in range(torch.cuda.device_count()):
                                        with torch.cuda.device(i):
                                            torch.cuda.empty_cache()
                                            torch.cuda.ipc_collect()
                                    
                            except (ImportError, AttributeError, Exception) as e:
                                logger.debug(f"Accelerate utilities not available or failed: {e}, using manual method")
                                
                                # Method 2: Manually move all modules to CPU
                                # This is more reliable for device_map models
                                def move_module_to_cpu(module):
                                    """Recursively move all modules to CPU"""
                                    for name, child in module.named_children():
                                        move_module_to_cpu(child)
                                    # Move this module's parameters and buffers
                                    for param in module.parameters(recurse=False):
                                        if param.is_cuda:
                                            param.data = param.data.cpu()
                                            if param.grad is not None and param.grad.is_cuda:
                                                param.grad = param.grad.cpu()
                                    for buffer in module.buffers(recurse=False):
                                        if buffer.is_cuda:
                                            buffer.data = buffer.data.cpu()
                                
                                # Move entire model tree to CPU
                                move_module_to_cpu(self.medgemma_model)
                                
                                # Also try to move model itself if it has a to method
                                try:
                                    self.medgemma_model = self.medgemma_model.to("cpu")
                                except:
                                    pass  # Some device_map models don't support .to()
                                
                                gc.collect()
                                if torch.cuda.is_available():
                                    for i in range(torch.cuda.device_count()):
                                        with torch.cuda.device(i):
                                            torch.cuda.empty_cache()
                                    
                        except Exception as e:
                            logger.warning(f"Could not move device_map model to CPU: {e}, trying parameter-by-parameter")
                            # Fallback: move parameters individually (most aggressive)
                            moved_count = 0
                            for name, param in self.medgemma_model.named_parameters():
                                if param.is_cuda:
                                    try:
                                        param.data = param.data.cpu()
                                        moved_count += 1
                                        if param.grad is not None and param.grad.is_cuda:
                                            param.grad = param.grad.cpu()
                                    except Exception as e2:
                                        logger.debug(f"Could not move parameter {name}: {e2}")
                            
                            for name, buffer in self.medgemma_model.named_buffers():
                                if buffer.is_cuda:
                                    try:
                                        buffer.data = buffer.data.cpu()
                                        moved_count += 1
                                    except Exception as e2:
                                        logger.debug(f"Could not move buffer {name}: {e2}")
                            
                            logger.info(f"Moved {moved_count} parameters/buffers to CPU")
                            
                            gc.collect()
                            if torch.cuda.is_available():
                                for i in range(torch.cuda.device_count()):
                                    with torch.cuda.device(i):
                                        for _ in range(3):
                                            torch.cuda.empty_cache()
                                            torch.cuda.ipc_collect()
                                        torch.cuda.synchronize()
                            
                except Exception as e:
                    logger.warning(f"Error during device_map cleanup: {e}")
            else:
                # Standard model - move to CPU
                try:
                    device = next(self.medgemma_model.parameters()).device
                    if device.type == 'cuda':
                        logger.info("Moving model to CPU before deletion")
                        self.medgemma_model = self.medgemma_model.to("cpu")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to move model to CPU: {e}")
            
            # Delete model and all references
            logger.info("Deleting model and processor...")
            
            # CRITICAL: For device_map models, we need to ensure all submodules are deleted
            if has_device_map:
                try:
                    # Clear all module references first
                    if hasattr(self.medgemma_model, 'modules'):
                        for module in list(self.medgemma_model.modules()):
                            try:
                                # Clear module's parameters and buffers
                                for param in list(module.parameters(recurse=False)):
                                    if param.is_cuda:
                                        param.data = None
                                        if param.grad is not None:
                                            param.grad = None
                                    del param
                                for buffer in list(module.buffers(recurse=False)):
                                    if buffer.is_cuda:
                                        buffer.data = None
                                    del buffer
                            except:
                                pass
                except Exception as e:
                    logger.debug(f"Error clearing module references: {e}")
            
            # Delete main model reference
            try:
                model_ref = self.medgemma_model
                self.medgemma_model = None
                del model_ref
            except:
                self.medgemma_model = None
            
            if self.medgemma_processor:
                try:
                    proc_ref = self.medgemma_processor
                    self.medgemma_processor = None
                    del proc_ref
                except:
                    self.medgemma_processor = None
                
            if self.medgemma_tokenizer:
                try:
                    tok_ref = self.medgemma_tokenizer
                    self.medgemma_tokenizer = None
                    del tok_ref
                except:
                    self.medgemma_tokenizer = None
            
            # Aggressive garbage collection - multiple passes
            logger.info("Running aggressive garbage collection...")
            for i in range(15):  # Increased from 10 to 15
                gc.collect()
            
            # Additional pass with explicit CUDA synchronization
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.synchronize()
                gc.collect()
            
            # Aggressive CUDA cache clearing
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache aggressively...")
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        # Get initial memory before cleanup
                        initial_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        initial_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        
                        # Multiple aggressive cache clears
                        for _ in range(10):  # Increased from 5 to 10
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                        
                        # Force memory release by allocating and freeing
                        # This helps CUDA's memory allocator release reserved memory
                        try:
                            # Allocate progressively larger tensors and free them
                            # This forces CUDA to consolidate and release memory
                            for size_mb in [100, 500, 1000, 2000, 5000]:
                                try:
                                    # Allocate tensor
                                    temp_tensor = torch.zeros(
                                        size_mb * 1024 * 1024 // 4,  # Approximate MB to elements (float32)
                                        dtype=torch.float32,
                                        device=f'cuda:{i}'
                                    )
                                    del temp_tensor
                                    torch.cuda.empty_cache()
                                except RuntimeError:
                                    # Out of memory - that's fine, continue
                                    break
                                except:
                                    pass
                        except:
                            pass
                        
                        # More cache clears after allocation test
                        for _ in range(5):
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        
                        torch.cuda.synchronize()
                        
                        # Reset peak stats
                        try:
                            torch.cuda.reset_peak_memory_stats(i)
                        except:
                            pass
                        
                        # Check final memory
                        final_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        final_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        
                        logger.info(
                            f"GPU {i} cache cleared: "
                            f"Allocated {initial_allocated:.2f}GB → {final_allocated:.2f}GB, "
                            f"Reserved {initial_reserved:.2f}GB → {final_reserved:.2f}GB"
                        )
                
                logger.info(f"GPU memory cache cleared on all devices")
            
            self.medgemma_loaded = False
            with self._reference_lock:
                self._reference_count = 0
            # Final verification - ensure memory is actually released
            if torch.cuda.is_available():
                import time
                time.sleep(0.5)  # Give GPU time to release memory
                
                # Multiple aggressive cache clears
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        # Clear cache multiple times
                        for _ in range(10):  # Increased from 1 to 10
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                        
                        # Force reserved memory release
                        try:
                            # Allocate and free tensors to trigger memory pool cleanup
                            for size in [100, 500, 1000]:
                                try:
                                    dummy = torch.zeros(size, size, device=f'cuda:{i}')
                                    del dummy
                                except:
                                    pass
                            torch.cuda.empty_cache()
                        except:
                            pass
                
                # One more garbage collection after cache clears
                for _ in range(5):
                    gc.collect()
                
                # Final cache clear
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
            
            # Verify model is actually deleted
            if self.medgemma_model is not None:
                logger.error("CRITICAL: Model still exists after deletion! Forcing removal...")
                self.medgemma_model = None
            
            logger.info("Medgemma model unloaded, GPU memory released")
        except Exception as e:
            logger.warning(f"Failed to release GPU memory: {e}", exc_info=True)
            # Still try to clear cache aggressively
            try:
                import torch
                import gc
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            for _ in range(5):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            torch.cuda.synchronize()
                for _ in range(3):
                    gc.collect()
            except:
                pass


class KnowledgeEngine:
    """Handle knowledge base search and retrieval
    Primary: Load FAQs from knowledge_base/sample_faqs.json
    Fallback: Use medgemma model for unanswered questions
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.faqs_data = None
        self.medgemma_model = None
        self.medgemma_processor = None
        self.medgemma_tokenizer = None
        self.medgemma_loaded = False  # Track if Medgemma is loaded
        self._model_manager = MedgemmaModelManager()  # Use shared model manager
        self._model_acquired = False  # Track if we've acquired model reference
        
        # CRITICAL: On initialization, check if model manager has inconsistent state
        # This can happen if process wasn't fully killed or cleanup failed
        try:
            if (hasattr(self._model_manager, 'medgemma_model') and 
                self._model_manager.medgemma_model is not None):
                logger.warning(
                    "Detected existing model in manager on KnowledgeEngine init. "
                    "This may indicate leftover state from previous run. Cleaning up..."
                )
                # Force cleanup to ensure clean state
                self._model_manager.cleanup(force=True)
                # Verify cleanup
                if self._model_manager.medgemma_model is not None:
                    logger.error("Model still exists after forced cleanup on init! This is a critical issue.")
                else:
                    logger.info("Successfully cleaned up leftover model state on init")
        except Exception as e:
            logger.warning(f"Error checking model manager state on init: {e}")
        
        # Initialize Redis for caching if available
        self.redis_client = None
        if REDIS_AVAILABLE and hasattr(settings, 'REDIS_URL'):
            try:
                self.redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=False)
                logger.info("Redis cache ready")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
        
        # Load FAQs from JSON file (primary source)
        self._load_faqs_from_json()
        
        # Use multilingual model that supports both English and Hindi
        try:
            self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Embedding model loaded: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to English-only model
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute FAQ embeddings for faster search (optional, can be disabled)
        if getattr(settings, 'PRE_COMPUTE_EMBEDDINGS', False):
            self._precompute_faq_embeddings()
        
        # Don't initialize medgemma on startup - use lazy loading
        # self._init_medgemma()  # Commented out for lazy loading
    
    def __del__(self):
        """Destructor to ensure cleanup when instance is garbage collected"""
        try:
            if self._model_acquired:
                self._release_gpu_memory()
        except:
            pass  # Ignore errors during cleanup
    
    def _load_faqs_from_json(self):
        """Load FAQs from knowledge_base/sample_faqs.json"""
        try:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent
            faqs_path = project_root / "knowledge_base" / "sample_faqs.json"
            
            if faqs_path.exists():
                with open(faqs_path, 'r', encoding='utf-8') as f:
                    self.faqs_data = json.load(f)
                logger.info(f"Loaded {len(self.faqs_data)} FAQs")
            else:
                logger.warning(f"FAQ file not found: {faqs_path}")
                self.faqs_data = []
        except Exception as e:
            logger.error(f"Failed to load FAQs: {e}")
            self.faqs_data = []
    
    def _precompute_faq_embeddings(self):
        """Pre-compute and cache FAQ embeddings for faster search"""
        if not self.faqs_data or not self.redis_client:
            return
        
        logger.info("Pre-computing FAQ embeddings...")
        try:
            # Collect all FAQ contents
            faq_contents = []
            for faq_data in self.faqs_data:
                # English content (optimized string concatenation)
                question_en = faq_data.get('question', '')
                answer_en = faq_data.get('answer', '')
                if question_en and answer_en:
                    faq_contents.append(' '.join((question_en, answer_en)))
                
                # Hindi + Hinglish content (so Hinglish queries match)
                question_hi = faq_data.get('question_hi', '')
                answer_hi = faq_data.get('answer_hi', '')
                question_hinglish = faq_data.get('question_hinglish', '')
                if question_hi and answer_hi:
                    parts = [question_hi, answer_hi]
                    if question_hinglish:
                        parts.insert(1, question_hinglish)
                    faq_contents.append(' '.join(parts))
            
            # Generate embeddings in batch
            if faq_contents:
                logger.info(f"Generating embeddings for {len(faq_contents)} FAQ contents...")
                self._get_embeddings_batch(faq_contents)
                logger.info("FAQ embeddings pre-computed and cached")
        except Exception as e:
            logger.warning(f"Failed to pre-compute embeddings: {e}")
    
    def _ensure_medgemma_loaded(self):
        """Lazy load Medgemma model only when needed (uses shared singleton)"""
        if self.medgemma_loaded and self.medgemma_model:
            return
        
        if not settings.USE_MEDGEMMA:
            logger.info("Medgemma disabled")
            return
        
        # Get model from shared manager
        model, processor, tokenizer = self._model_manager.get_model()
        if model:
            self.medgemma_model = model
            self.medgemma_processor = processor
            self.medgemma_tokenizer = tokenizer
            self.medgemma_loaded = True
            self._model_acquired = True
        else:
            self.medgemma_loaded = False
    
    def _release_gpu_memory(self):
        """Release model reference (model stays in memory for reuse, only cleanup if needed)"""
        if not self._model_acquired:
            return
        
        # Release reference to shared model
        self._model_manager.release_model()
        
        # Clear local references
        self.medgemma_model = None
        self.medgemma_processor = None
        self.medgemma_tokenizer = None
        self.medgemma_loaded = False
        self._model_acquired = False
        
        # Note: We don't unload the model here - it stays in memory for reuse
        # The model manager will handle cleanup when appropriate
        logger.debug("Released model reference (model remains in memory for reuse)")
    
    def _get_gpu_memory_stats(self):
        """Get current GPU memory statistics for all devices"""
        import torch
        stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                    stats[i] = {
                        'allocated_gb': allocated,
                        'reserved_gb': reserved,
                        'max_allocated_gb': max_allocated
                    }
        return stats
    
    def _force_release_reserved_memory(self, device_id: int = 0):
        """
        Force CUDA to release reserved memory by aggressive cleanup.
        This helps reduce the reserved memory pool that CUDA keeps.
        
        NOTE: CUDA's memory allocator keeps memory reserved even after PyTorch
        releases it. This method tries to force CUDA to shrink the memory pool,
        but some reserved memory may persist until the process ends.
        """
        import torch
        import time
        import gc
        
        if not torch.cuda.is_available():
            return
        
        try:
            with torch.cuda.device(device_id):
                # Get initial reserved memory
                initial_reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
                initial_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                
                if initial_reserved < 0.5:  # Less than 500MB, no need to force release
                    return
                
                logger.info(f"Forcing release of reserved memory (current: {initial_reserved:.2f}GB reserved, {initial_allocated:.2f}GB allocated)...")
                
                # Step 1: Multiple aggressive cache clears with delays
                for _ in range(15):  # Increased from 10
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    time.sleep(0.02)
                
                # Step 2: Try to trigger CUDA memory pool shrinkage by allocating/freeing
                # CUDA's allocator may shrink the pool if we allocate and free memory
                try:
                    # Allocate progressively larger chunks and free them
                    # This can help CUDA's allocator consolidate and release memory
                    allocations = []
                    for size_mb in [50, 100, 200, 500, 1000]:
                        try:
                            # Calculate elements for approximate MB (float32 = 4 bytes)
                            elements = (size_mb * 1024 * 1024) // 4
                            tensor = torch.zeros(elements, dtype=torch.float32, device=f'cuda:{device_id}')
                            allocations.append(tensor)
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            # Out of memory - that's expected, continue
                            break
                        except:
                            pass
                    
                    # Free all allocations
                    for tensor in allocations:
                        del tensor
                    allocations.clear()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Error during allocation test: {e}")
                
                # Step 3: More aggressive clearing with synchronization
                for _ in range(10):  # Increased from 5
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                    time.sleep(0.05)
                
                # Step 4: Reset peak stats to help with memory tracking
                try:
                    torch.cuda.reset_peak_memory_stats(device_id)
                except:
                    pass
                
                # Step 5: Final aggressive clear
                for _ in range(5):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                
                # Check final reserved memory
                final_reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB
                final_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                released_reserved = initial_reserved - final_reserved
                released_allocated = initial_allocated - final_allocated
                
                logger.info(
                    f"Reserved memory: {initial_reserved:.2f}GB → {final_reserved:.2f}GB "
                    f"({released_reserved:+.2f}GB released)"
                )
                logger.info(
                    f"Allocated memory: {initial_allocated:.2f}GB → {final_allocated:.2f}GB "
                    f"({released_allocated:+.2f}GB released)"
                )
                
                # Note about CUDA memory pool behavior
                if final_reserved > 1.0:
                    logger.warning(
                        f"Reserved memory still high ({final_reserved:.2f}GB). "
                        f"This is CUDA's memory pool - it may not release until process ends. "
                        f"However, this memory is available for reuse and won't cause leaks."
                    )
                
        except Exception as e:
            logger.debug(f"Error forcing reserved memory release: {e}")
    
    def _log_gpu_memory_stats(self, stage: str):
        """Log GPU memory statistics at a given stage"""
        try:
            stats = self._get_gpu_memory_stats()
            if stats:
                for device_id, mem_info in stats.items():
                    logger.info(
                        f"GPU {device_id} Memory [{stage}]: "
                        f"Allocated={mem_info['allocated_gb']:.2f}GB, "
                        f"Reserved={mem_info['reserved_gb']:.2f}GB, "
                        f"Peak={mem_info['max_allocated_gb']:.2f}GB"
                    )
        except Exception as e:
            logger.debug(f"Could not get GPU memory stats: {e}")
    
    def _clear_gpu_memory_before_inference(self):
        """
        Clear GPU memory before starting inference to ensure clean state.
        This prevents memory accumulation from previous requests.
        """
        import torch
        import gc
        
        try:
            logger.info("Clearing GPU memory before inference...")
            
            # Log initial memory state
            self._log_gpu_memory_stats("BEFORE CLEARING")
            
            # Step 1: Aggressive garbage collection
            for _ in range(5):  # Increased from 3 to 5
                gc.collect()
            
            # Step 2: Clear CUDA cache on all devices aggressively
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            # Clear cache multiple times
                            for _ in range(5):  # Increased from 2 to 5
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            
                            # Force reserved memory release
                            try:
                                dummy = torch.zeros(1, device=f'cuda:{i}')
                                del dummy
                                torch.cuda.empty_cache()
                            except:
                                pass
                            
                            torch.cuda.synchronize()
                    logger.debug("CUDA cache cleared on all devices before inference")
                except Exception as e:
                    logger.warning(f"Error clearing CUDA cache before inference: {e}")
            
            # Log memory after clearing
            self._log_gpu_memory_stats("AFTER CLEARING")
            
            logger.info("GPU memory cleared before inference")
            
        except Exception as e:
            logger.warning(f"Error during pre-inference GPU memory clear: {e}")
            # Continue anyway - this is a best-effort cleanup
    
    def _release_gpu_memory_after_inference(self):
        """
        Release GPU memory immediately after inference - AGGRESSIVE cleanup
        
        This method ensures GPU memory is ALWAYS released after each Medgemma response,
        even if errors occur during cleanup. It performs:
        1. Clears local model references
        2. Forces model manager cleanup to unload model from GPU
        3. Aggressive CUDA cache clearing
        4. Multiple garbage collection passes
        """
        import torch
        import gc
        
        try:
            # Log memory before cleanup
            self._log_gpu_memory_stats("BEFORE CLEANUP")
            
            logger.info("Releasing GPU memory after MedGemma response...")
            
            # Step 1: Release local references
            self._release_gpu_memory()
            
            # Step 2: Force cleanup of the model to unload from GPU
            # CRITICAL: ALWAYS unload model after each response to release GPU memory
            # This ensures memory is freed even if the model manager wants to keep it for reuse
            try:
                # First, reset reference count to allow cleanup
                with self._model_manager._reference_lock:
                    self._model_manager._reference_count = 0
                
                # Force cleanup - this should unload the model
                self._model_manager.cleanup(force=True)
                
                # CRITICAL: Verify and force unload if model still exists
                # Even if cleanup() was called, we must ensure model is actually unloaded
                if (hasattr(self._model_manager, 'medgemma_model') and 
                    self._model_manager.medgemma_model is not None):
                    logger.warning("Model still exists after cleanup(force=True), forcing aggressive unload...")
                    # Force direct cleanup - move to CPU and delete
                    try:
                        import torch
                        model = self._model_manager.medgemma_model
                        
                        # For device_map models, we need special handling
                        has_device_map = (hasattr(model, 'hf_device_map') or 
                                        hasattr(model, 'device_map'))
                        
                        if has_device_map:
                            # Device_map models: move all modules to CPU individually
                            logger.info("Unloading device_map model module-by-module...")
                            try:
                                # Get all device mappings
                                if hasattr(model, 'hf_device_map'):
                                    device_map = model.hf_device_map
                                else:
                                    device_map = model.device_map
                                
                                # Move each module to CPU
                                for module_name, device in device_map.items():
                                    if device != 'cpu':
                                        try:
                                            module = model.get_submodule(module_name) if hasattr(model, 'get_submodule') else None
                                            if module is None:
                                                # Try alternative way to get module
                                                parts = module_name.split('.')
                                                module = model
                                                for part in parts:
                                                    module = getattr(module, part, None)
                                                    if module is None:
                                                        break
                                            
                                            if module is not None and hasattr(module, 'to'):
                                                module.to('cpu')
                                        except Exception as e:
                                            logger.debug(f"Could not move {module_name} to CPU: {e}")
                            except Exception as e:
                                logger.warning(f"Error moving device_map model to CPU: {e}")
                        else:
                            # Standard model: move entire model to CPU
                            if hasattr(model, 'to'):
                                try:
                                    model = model.to('cpu')
                                except Exception as e:
                                    logger.warning(f"Could not move model to CPU: {e}")
                        
                        # Delete the model
                        del model
                        self._model_manager.medgemma_model = None
                        self._model_manager.medgemma_processor = None
                        self._model_manager.medgemma_tokenizer = None
                        self._model_manager.medgemma_loaded = False
                        
                        # Clear CUDA cache
                        if torch.cuda.is_available():
                            for _ in range(5):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            torch.cuda.synchronize()
                        
                        logger.info("Model forcefully unloaded from GPU")
                    except Exception as e2:
                        logger.error(f"Failed to force unload model: {e2}", exc_info=True)
                        # Last resort: just set to None even if deletion failed
                        self._model_manager.medgemma_model = None
                        self._model_manager.medgemma_processor = None
                        self._model_manager.medgemma_tokenizer = None
                        self._model_manager.medgemma_loaded = False
                else:
                    logger.debug("Model successfully unloaded by cleanup()")
            except Exception as e:
                logger.error(f"Error during model manager cleanup: {e}", exc_info=True)
                # Try direct cleanup as fallback - ALWAYS ensure model is unloaded
                try:
                    if hasattr(self._model_manager, 'medgemma_model') and self._model_manager.medgemma_model is not None:
                        logger.warning("Attempting emergency direct model cleanup...")
                        import torch
                        model = self._model_manager.medgemma_model
                        
                        # Try to move to CPU
                        if hasattr(model, 'to'):
                            try:
                                model = model.to('cpu')
                            except:
                                pass
                        
                        # Delete model
                        del model
                        self._model_manager.medgemma_model = None
                        self._model_manager.medgemma_processor = None
                        self._model_manager.medgemma_tokenizer = None
                        self._model_manager.medgemma_loaded = False
                        
                        # Reset reference count
                        with self._model_manager._reference_lock:
                            self._model_manager._reference_count = 0
                        
                        # Aggressive cache clearing
                        if torch.cuda.is_available():
                            for _ in range(5):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            torch.cuda.synchronize()
                        
                        logger.warning("Emergency cleanup completed")
                except Exception as e2:
                    logger.error(f"Emergency cleanup also failed: {e2}", exc_info=True)
                    # Final fallback: just clear the references
                    try:
                        self._model_manager.medgemma_model = None
                        self._model_manager.medgemma_processor = None
                        self._model_manager.medgemma_tokenizer = None
                        self._model_manager.medgemma_loaded = False
                        with self._model_manager._reference_lock:
                            self._model_manager._reference_count = 0
                    except:
                        pass
            
            # Step 3: Aggressive garbage collection
            for _ in range(5):
                gc.collect()
            
            # Step 4: Aggressive CUDA cache clearing with reserved memory release
            # CRITICAL: This step ensures all GPU memory is actually freed
            if torch.cuda.is_available():
                try:
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            # Multiple aggressive cache clears (increased for better cleanup)
                            for _ in range(10):  # Increased from 5 to 10 for more thorough cleanup
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            
                            torch.cuda.synchronize()
                            
                            # Reset peak stats after cleanup
                            try:
                                torch.cuda.reset_peak_memory_stats(i)
                            except:
                                pass
                            
                            # Additional aggressive cleanup: allocate and free memory to force pool shrinkage
                            try:
                                # Try to allocate small tensors and free them to trigger memory pool cleanup
                                for size in [100, 500, 1000]:
                                    try:
                                        dummy = torch.zeros(size, size, device=f'cuda:{i}', dtype=torch.float16)
                                        del dummy
                                    except:
                                        pass
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            except:
                                pass
                            
                    # Force release of reserved memory on all devices
                    for i in range(torch.cuda.device_count()):
                        self._force_release_reserved_memory(i)
                    
                    logger.debug("CUDA cache cleared on all devices (with reserved memory release)")
                except Exception as e:
                    logger.warning(f"Error clearing CUDA cache: {e}")
            
            # Log memory after cleanup
            self._log_gpu_memory_stats("AFTER CLEANUP")
            
            # CRITICAL: Verify cleanup actually worked
            # This verification ensures memory is actually released after each response
            try:
                stats = self._get_gpu_memory_stats()
                if stats:
                    for device_id, mem_info in stats.items():
                        allocated = mem_info['allocated_gb']
                        reserved = mem_info['reserved_gb']
                        
                        # If allocated memory is still high (>1GB), something went wrong
                        # Lowered threshold from 2GB to 1GB for stricter verification
                        if allocated > 1.0:
                            logger.error(
                                f"CRITICAL: GPU {device_id} allocated memory still high after cleanup: "
                                f"{allocated:.2f}GB. Model may not have been unloaded properly! "
                                f"Attempting emergency cleanup..."
                            )
                            # Try emergency cleanup - more aggressive
                            try:
                                import torch
                                import gc
                                
                                # Verify model is actually gone
                                if hasattr(self._model_manager, 'medgemma_model') and self._model_manager.medgemma_model:
                                    logger.error("EMERGENCY: Model still exists in manager! Forcing deletion...")
                                    model = self._model_manager.medgemma_model
                                    
                                    # Try to move to CPU
                                    if hasattr(model, 'to'):
                                        try:
                                            model = model.to('cpu')
                                        except:
                                            pass
                                    
                                    # Delete model
                                    del model
                                    self._model_manager.medgemma_model = None
                                    self._model_manager.medgemma_processor = None
                                    self._model_manager.medgemma_tokenizer = None
                                    self._model_manager.medgemma_loaded = False
                                    
                                    # Reset reference count
                                    with self._model_manager._reference_lock:
                                        self._model_manager._reference_count = 0
                                
                                # Aggressive garbage collection
                                for _ in range(15):  # Increased from 10 to 15
                                    gc.collect()
                                
                                # Aggressive CUDA cache clearing
                                if torch.cuda.is_available():
                                    for i in range(torch.cuda.device_count()):
                                        with torch.cuda.device(i):
                                            for _ in range(15):  # Increased from 10 to 15
                                                torch.cuda.empty_cache()
                                                torch.cuda.ipc_collect()
                                            torch.cuda.synchronize()
                                            
                                            # Try to force memory pool shrinkage
                                            try:
                                                for size in [100, 500, 1000, 2000]:
                                                    try:
                                                        dummy = torch.zeros(size, size, device=f'cuda:{i}', dtype=torch.float16)
                                                        del dummy
                                                    except:
                                                        pass
                                                torch.cuda.empty_cache()
                                                torch.cuda.ipc_collect()
                                            except:
                                                pass
                                
                                # Check again after emergency cleanup
                                stats_after = self._get_gpu_memory_stats()
                                if stats_after:
                                    for dev_id, mem_info_after in stats_after.items():
                                        allocated_after = mem_info_after['allocated_gb']
                                        reserved_after = mem_info_after['reserved_gb']
                                        
                                        logger.info(
                                            f"GPU {dev_id} Memory [AFTER EMERGENCY CLEANUP]: "
                                            f"Allocated={allocated_after:.2f}GB, "
                                            f"Reserved={reserved_after:.2f}GB"
                                        )
                                        
                                        if allocated_after > 1.0:
                                            logger.error(
                                                f"CRITICAL: GPU {dev_id} memory still high after emergency cleanup! "
                                                f"This may indicate a memory leak or CUDA memory pool issue."
                                            )
                                        else:
                                            logger.info(f"GPU {dev_id} memory successfully released after emergency cleanup")
                            except Exception as e:
                                logger.error(f"Emergency cleanup failed: {e}", exc_info=True)
                        else:
                            logger.info(
                                f"✓ GPU {device_id} cleanup verified: "
                                f"Allocated={allocated:.2f}GB (target: <1GB), "
                                f"Reserved={reserved:.2f}GB"
                            )
            except Exception as e:
                logger.debug(f"Could not verify cleanup: {e}")
            
            logger.info("GPU memory released successfully")
            
        except Exception as e:
            logger.error(f"Critical error during GPU memory release: {e}", exc_info=True)
            # Even if cleanup fails, try to clear CUDA cache as last resort
            try:
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            # Multiple aggressive clears
                            for _ in range(5):
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            torch.cuda.synchronize()
                            # Try to force reserved memory release
                            try:
                                dummy = torch.zeros(1, device=f'cuda:{i}')
                                del dummy
                                torch.cuda.empty_cache()
                            except:
                                pass
                    for _ in range(5):
                        gc.collect()
                # Log memory even after error
                self._log_gpu_memory_stats("AFTER CLEANUP (ERROR)")
            except:
                pass
    
    def cleanup(self, release_model: bool = False):
        """Cleanup method to release all resources including GPU memory"""
        logger.info("Cleaning up KnowledgeEngine resources...")
        self._release_gpu_memory()
        
        # Optionally force cleanup of shared model (only if explicitly requested)
        if release_model:
            self._model_manager.cleanup(force=True)
        
        # Also cleanup encoder if needed
        if hasattr(self, 'encoder'):
            try:
                del self.encoder
            except:
                pass
        
        # Close Redis connection if exists
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
    
    def _init_medgemma(self):
        """Initialize medgemma multimodal model for fallback responses (deprecated - use _ensure_medgemma_loaded)"""
        # This method is kept for backward compatibility but now uses the shared manager
        self._ensure_medgemma_loaded()
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (optimized with caching)"""
        # Check cache first
        if self.redis_client:
            cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        
        try:
            # Use batch encoding for better performance
            embedding = self.encoder.encode(text, convert_to_numpy=True, show_progress_bar=False, batch_size=1)
            embedding_list = embedding.tolist()
            
            # Cache embedding
            if self.redis_client:
                try:
                    cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
                    self.redis_client.setex(cache_key, 86400, json.dumps(embedding_list))  # Cache for 24 hours
                except:
                    pass
            
            return embedding_list
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []
    
    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch (much faster)"""
        if not texts:
            return []
        
        # Check cache for all texts
        cached_embeddings = {}
        texts_to_encode = []
        text_indices = []
        
        if self.redis_client:
            for i, text in enumerate(texts):
                cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
                try:
                    cached = self.redis_client.get(cache_key)
                    if cached:
                        cached_embeddings[i] = json.loads(cached)
                    else:
                        texts_to_encode.append(text)
                        text_indices.append(i)
                except:
                    texts_to_encode.append(text)
                    text_indices.append(i)
        else:
            texts_to_encode = texts
            text_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts in batch
        if texts_to_encode:
            try:
                embeddings = self.encoder.encode(
                    texts_to_encode,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(32, len(texts_to_encode))  # Batch size for efficiency
                )
                
                # Cache new embeddings
                if self.redis_client:
                    for idx, text in zip(text_indices, texts_to_encode):
                        try:
                            cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
                            self.redis_client.setex(cache_key, 86400, json.dumps(embeddings[idx].tolist()))
                        except:
                            pass
                
                # Combine cached and new embeddings
                result = []
                cached_idx = 0
                encode_idx = 0
                for i in range(len(texts)):
                    if i in cached_embeddings:
                        result.append(cached_embeddings[i])
                    else:
                        result.append(embeddings[encode_idx].tolist())
                        encode_idx += 1
                return result
            except Exception as e:
                logger.error(f"Batch embedding generation failed: {e}")
                # Fallback to individual encoding
                return [self._get_embedding(text) for text in texts]
        
        # Return all cached embeddings
        return [cached_embeddings[i] for i in range(len(texts))]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors (optimized)"""
        try:
            vec1 = np.array(vec1, dtype=np.float32)  # Use float32 for memory efficiency
            vec2 = np.array(vec2, dtype=np.float32)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _cosine_similarity_batch(self, query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and multiple documents (vectorized)
        Time complexity: O(n*d) where n=num docs, d=embedding dim (much faster than O(n*d) loops)
        """
        try:
            # Normalize query vector once
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return np.zeros(len(doc_vecs))
            
            # Vectorized dot product: O(n*d) - single operation instead of n operations
            dot_products = np.dot(doc_vecs, query_vec)
            
            # Vectorized norm calculation: O(n*d) - single operation
            doc_norms = np.linalg.norm(doc_vecs, axis=1)
            
            # Avoid division by zero
            doc_norms = np.where(doc_norms == 0, 1.0, doc_norms)
            
            # Vectorized division: O(n)
            similarities = dot_products / (doc_norms * query_norm)
            
            return similarities.astype(np.float32)
        except Exception as e:
            logger.error(f"Batch similarity calculation failed: {e}")
            return np.zeros(len(doc_vecs))
    
    def _get_cache_key(self, query: str, language: str, cache_type: str = "faq") -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(f"{cache_type}:{query}:{language}".encode()).hexdigest()
        return f"knowledge_cache:{query_hash}"

    def _normalize_query_for_search(self, query: str, language: str) -> str:
        """
        For Hindi search, ensure the English term IVF is aligned with आईवीएफ
        so semantic search matches Hindi content that uses आईवीएफ.
        """
        if not query or not query.strip() or language != "hi":
            return query
        q = query.strip()
        if re.search(r"\bivf\b", q, re.IGNORECASE):
            return q + " आईवीएफ"
        return query
    
    def _get_cached_response(self, cache_key: str) -> Optional[List[Dict]]:
        """Get cached response from Redis"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: List[Dict], ttl: int = 3600):
        """Cache response in Redis"""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(response, default=str)
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def search_faqs(
        self,
        query: str,
        language: str = "en",
        top_k: int = 5,
        category: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Search FAQs by semantic similarity
        Primary: Search in JSON file (knowledge_base/sample_faqs.json)
        Secondary: Search in database if JSON doesn't have enough results
        
        Args:
            query: Search query
            language: Language code ('en' or 'hi')
            top_k: Number of results to return
            category: Optional category filter
            use_cache: Whether to use cached results
        
        Returns:
            List of matching FAQs with similarity scores
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(query, language, f"faq_{category or 'all'}")
            cached_results = self._get_cached_response(cache_key)
            if cached_results is not None:
                logger.debug(f"FAQ cache hit: {query[:50]}")
                return cached_results[:top_k]
        
        results = []
        
        # PRIMARY: Search in JSON file (optimized with batch embedding)
        if self.faqs_data:
            query_embedding = self._get_embedding(query)
            if query_embedding:
                # Prepare all FAQ contents for batch processing
                faq_contents = []
                faq_metadata = []
                
                for faq_data in self.faqs_data:
                    # Apply category filter if specified
                    if category and faq_data.get('category') != category:
                        continue
                    
                    # Get content in requested language (include Hinglish for Hindi so Roman-script queries match)
                    if language == 'hi':
                        question = faq_data.get('question_hi', faq_data.get('question', ''))
                        answer = faq_data.get('answer_hi', faq_data.get('answer', ''))
                        question_hinglish = faq_data.get('question_hinglish', '')
                        content = ' '.join(filter(None, [question, question_hinglish, answer]))
                    else:
                        question = faq_data.get('question', '')
                        answer = faq_data.get('answer', '')
                        content = ' '.join((question, answer))
                    
                    if not question or not answer:
                        continue
                    faq_contents.append(content)
                    faq_metadata.append({
                        'faq_id': f"json_{hash(question)}",
                        'question': question,
                        'answer': answer,
                        'category': faq_data.get('category'),
                        'tags': faq_data.get('tags', [])
                    })
                
                # Generate all embeddings in batch (much faster)
                if faq_contents:
                    faq_embeddings = self._get_embeddings_batch(faq_contents)
                    
                    # Filter out None embeddings and prepare for vectorized calculation
                    valid_indices = []
                    valid_embeddings = []
                    valid_metadata = []
                    
                    for i, (faq_embedding, metadata) in enumerate(zip(faq_embeddings, faq_metadata)):
                        if faq_embedding:
                            valid_indices.append(i)
                            valid_embeddings.append(faq_embedding)
                            valid_metadata.append(metadata)
                    
                    if valid_embeddings:
                        # Vectorized similarity calculation: O(n*d) instead of O(n*d) loops
                        query_vec = np.array(query_embedding, dtype=np.float32)
                        doc_vecs = np.array(valid_embeddings, dtype=np.float32)
                        similarities = self._cosine_similarity_batch(query_vec, doc_vecs)
                        
                        # Build results in one pass: O(n)
                        for similarity, metadata in zip(similarities, valid_metadata):
                            results.append({
                                'faq_id': metadata['faq_id'],
                                'question': metadata['question'],
                                'answer': metadata['answer'],
                                'category': metadata['category'],
                                'tags': metadata['tags'],
                                'similarity': float(similarity),
                                'type': 'faq',
                                'source': 'json'
                            })
        
        # SECONDARY: Search in database if we need more results
        if len(results) < top_k:
            try:
                query_obj = self.db.query(FAQ)
                if category:
                    query_obj = query_obj.filter(FAQ.category == category)
                
                db_faqs = query_obj.all()
                
                if not query_embedding:
                    query_embedding = self._get_embedding(query)
                
                # Batch process database FAQs for better performance
                db_contents_to_embed = []
                db_embeddings = []
                db_metadata = []
                db_indices = []
                
                for i, faq in enumerate(db_faqs):
                    # Get content in requested language (optimized string concatenation)
                    if language == 'hi' and faq.question_hi and faq.answer_hi:
                        content = ' '.join((faq.question_hi, faq.answer_hi))
                        question = faq.question_hi
                        answer = faq.answer_hi
                    else:
                        content = ' '.join((faq.question, faq.answer))
                        question = faq.question
                        answer = faq.answer
                    
                    # Check for cached embedding
                    if faq.embedding:
                        try:
                            faq_embedding = json.loads(faq.embedding)
                            db_embeddings.append(faq_embedding)
                            db_indices.append(i)
                        except:
                            db_contents_to_embed.append(content)
                            db_indices.append(i)
                    else:
                        db_contents_to_embed.append(content)
                        db_indices.append(i)
                    
                    db_metadata.append({
                        'faq_id': str(faq.faq_id),
                        'question': question,
                        'answer': answer,
                        'category': faq.category,
                        'tags': faq.tags or []
                    })
                
                # Batch generate embeddings for FAQs without cached embeddings
                if db_contents_to_embed:
                    new_embeddings = self._get_embeddings_batch(db_contents_to_embed)
                    for embedding in new_embeddings:
                        if embedding:
                            db_embeddings.append(embedding)
                
                # Vectorized similarity calculation for database FAQs
                if db_embeddings and len(db_embeddings) == len(db_indices):
                    query_vec = np.array(query_embedding, dtype=np.float32)
                    doc_vecs = np.array(db_embeddings, dtype=np.float32)
                    similarities = self._cosine_similarity_batch(query_vec, doc_vecs)
                    
                    for similarity, idx in zip(similarities, db_indices):
                        metadata = db_metadata[idx]
                        results.append({
                            'faq_id': metadata['faq_id'],
                            'question': metadata['question'],
                            'answer': metadata['answer'],
                            'category': metadata['category'],
                            'tags': metadata['tags'],
                            'similarity': float(similarity),
                            'type': 'faq',
                            'source': 'database'
                        })
            except Exception as e:
                logger.error(f"Database FAQ search failed: {e}")
        
        # Sort by similarity and return top_k (optimized: use partial sort for large lists)
        if len(results) > top_k * 2:
            # For large result sets, use partial sort: O(n + k*log(k)) instead of O(n*log(n))
            import heapq
            final_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity'])
        else:
            # For small result sets, full sort is faster: O(n*log(n))
            results.sort(key=lambda x: x['similarity'], reverse=True)
            final_results = results[:top_k]
        
        # Log search results for debugging
        if final_results:
            logger.debug(f"FAQ search: {len(final_results)} results (best: {final_results[0]['similarity']:.2f})")
        else:
            logger.debug(f"FAQ search: no results for '{query[:50]}'")
        
        # Cache results
        if use_cache:
            cache_key = self._get_cache_key(query, language, f"faq_{category or 'all'}")
            cache_ttl = getattr(settings, 'CACHE_TTL_FAQ', 3600)  # Default 1 hour
            self._cache_response(cache_key, final_results, ttl=cache_ttl)
        
        return final_results
    
    def search_articles(
        self,
        query: str,
        language: str = "en",
        top_k: int = 5,
        category: Optional[str] = None
    ) -> List[Dict]:
        """Search articles by semantic similarity (optimized with batch processing)"""
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            return []
        
        query_obj = self.db.query(Article)
        if category:
            query_obj = query_obj.filter(Article.category == category)
        
        articles = query_obj.all()
        
        if not articles:
            return []
        
        # Batch process articles for better performance
        article_contents = []
        article_metadata = []
        articles_to_embed = []
        embed_indices = []
        
        for i, article in enumerate(articles):
            # Optimize string operations
            if language == 'hi' and article.title_hi and article.content_hi:
                content = ' '.join((article.title_hi, article.content_hi))
                title = article.title_hi
                content_text = article.content_hi
            else:
                content = ' '.join((article.title, article.content))
                title = article.title
                content_text = article.content
            
            # Check for cached embedding
            if article.embedding:
                try:
                    article_embedding = json.loads(article.embedding)
                    article_contents.append((article_embedding, i))
                except:
                    articles_to_embed.append(content)
                    embed_indices.append(i)
            else:
                articles_to_embed.append(content)
                embed_indices.append(i)
            
            article_metadata.append({
                'article_id': str(article.article_id),
                'title': title,
                'content': content_text,
                'category': article.category,
                'tags': article.tags or []
            })
        
        # Batch generate embeddings for articles without cached embeddings
        if articles_to_embed:
            new_embeddings = self._get_embeddings_batch(articles_to_embed)
            for embedding, idx in zip(new_embeddings, embed_indices):
                if embedding:
                    article_contents.append((embedding, idx))
        
        # Vectorized similarity calculation
        if article_contents:
            embeddings_list = [emb for emb, _ in article_contents]
            indices_list = [idx for _, idx in article_contents]
            
            query_vec = np.array(query_embedding, dtype=np.float32)
            doc_vecs = np.array(embeddings_list, dtype=np.float32)
            similarities = self._cosine_similarity_batch(query_vec, doc_vecs)
            
            results = []
            for similarity, idx in zip(similarities, indices_list):
                metadata = article_metadata[idx]
                results.append({
                    'article_id': metadata['article_id'],
                    'title': metadata['title'],
                    'content': metadata['content'],
                    'category': metadata['category'],
                    'tags': metadata['tags'],
                    'similarity': float(similarity),
                    'type': 'article'
                })
            
            # Optimized sorting: use partial sort for large lists
            if len(results) > top_k * 2:
                import heapq
                return heapq.nlargest(top_k, results, key=lambda x: x['similarity'])
            else:
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:top_k]
        
        return []
    
    def search(
        self,
        query: str,
        language: str = "en",
        top_k: int = 5,
        content_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Unified search across FAQs and articles
        
        Args:
            query: Search query
            language: Language code
            top_k: Number of results
            content_types: List of content types to search ('faq', 'article')
        
        Returns:
            Combined and ranked results
        """
        if content_types is None:
            content_types = ['faq', 'article']
        
        # For Hindi, treat "IVF" as आईवीएफ so search matches Hindi FAQ/content
        query = self._normalize_query_for_search(query, language)
        all_results = []
        
        if 'faq' in content_types:
            faq_results = self.search_faqs(query, language, top_k)
            all_results.extend(faq_results)
        
        if 'article' in content_types:
            article_results = self.search_articles(query, language, top_k)
            all_results.extend(article_results)
        
        # Sort all results by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return all_results[:top_k]
    
    def get_faq_by_id(self, faq_id: str, language: str = "en") -> Optional[Dict]:
        """Get FAQ by ID"""
        # Check JSON file first
        if self.faqs_data and faq_id.startswith("json_"):
            # Extract index from hash-based ID
            for faq_data in self.faqs_data:
                question = faq_data.get('question', '')
                if f"json_{hash(question)}" == faq_id:
                    if language == 'hi':
                        return {
                            'faq_id': faq_id,
                            'question': faq_data.get('question_hi', faq_data.get('question', '')),
                            'answer': faq_data.get('answer_hi', faq_data.get('answer', '')),
                            'category': faq_data.get('category'),
                            'tags': faq_data.get('tags', [])
                        }
                    else:
                        return {
                            'faq_id': faq_id,
                            'question': faq_data.get('question', ''),
                            'answer': faq_data.get('answer', ''),
                            'category': faq_data.get('category'),
                            'tags': faq_data.get('tags', [])
                        }
        
        # Check database
        from uuid import UUID
        try:
            faq = self.db.query(FAQ).filter(FAQ.faq_id == UUID(faq_id)).first()
            if faq:
                return {
                    'faq_id': str(faq.faq_id),
                    'question': faq.question_hi if language == 'hi' and faq.question_hi else faq.question,
                    'answer': faq.answer_hi if language == 'hi' and faq.answer_hi else faq.answer,
                    'category': faq.category,
                    'tags': faq.tags or []
                }
        except Exception as e:
            logger.error(f"Failed to get FAQ: {e}")
        return None
    
    def _get_medgemma_cached_answer(self, query: str, language: str, context: Optional[str]) -> Optional[str]:
        """Check cache for Medgemma answer"""
        if not self.redis_client:
            return None
        
        cache_key = self._get_cache_key(query, language, f"medgemma_{hash(context or '')}")
        try:
            cached_answer = self.redis_client.get(cache_key)
            if cached_answer:
                logger.debug(f"Medgemma cache hit: {query[:50]}")
                if isinstance(cached_answer, bytes):
                    return cached_answer.decode('utf-8')
                return cached_answer
        except Exception as e:
            logger.warning(f"Medgemma cache read failed: {e}")
        return None
    
    def _prepare_medgemma_prompt(self, query: str, context: Optional[str]) -> Tuple[str, str]:
        """Prepare question text and system prompt for Medgemma"""
        # Normalize ivf/IVF/आईवीएफ in query so model treats them as the same (IVF)
        query_normalized = re.sub(r"\bivf\b", "IVF", query, flags=re.IGNORECASE)
        # Direct question only – avoid long wrappers that encourage meta-replies
        question_text = query_normalized.strip()
        if context:
            context_normalized = re.sub(r"\bivf\b", "IVF", context, flags=re.IGNORECASE)
            question_text = f"Context: {context_normalized}\n\nQuestion: {query_normalized.strip()}"
        
        system_prompt = (
            "You are an IVF (In Vitro Fertilization) advisor. Answer ONLY the user's question. "
            "Do NOT repeat these instructions, say you are ready to help, or list what you will do. "
            "Give a direct, useful answer only. 'ivf', 'IVF', and 'आईवीएफ' mean the same.\n\n"
            "Rules: (1) Answer only IVF-related questions; if off-topic, briefly decline. "
            "(2) For cost, price, expense, or kharcha: give amounts in Indian Rupees (₹ / INR) only, never US dollars. "
            "(3) Be concise, factual, and supportive. For personal medical decisions, suggest consulting a specialist."
        )
        return question_text, system_prompt
    
    def _prepare_medgemma_inputs(
        self,
        question_text: str,
        system_prompt: str,
        image: Optional[Any] = None,
    ) -> Tuple[Dict, int]:
        """Prepare and process inputs for Medgemma model (text-only or multimodal with image)."""
        import torch

        if image is not None and PIL_AVAILABLE and hasattr(image, "convert"):
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": question_text},
            ]
            logger.debug("MedGemma multimodal: image + text")
        else:
            user_content = [{"type": "text", "text": question_text}]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        inputs = self.medgemma_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = next(self.medgemma_model.parameters()).device
        dtype = next(self.medgemma_model.parameters()).dtype

        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype in [torch.int64, torch.int32, torch.long, torch.int]:
                    processed_inputs[k] = v.to(device)
                else:
                    processed_inputs[k] = v.to(device, dtype=dtype)
            else:
                processed_inputs[k] = v

        input_len = processed_inputs["input_ids"].shape[-1]
        logger.debug(f"Input tokens: {input_len}, device: {device}, keys: {list(processed_inputs.keys())}")
        return processed_inputs, input_len
    
    def _generate_medgemma_output(self, inputs: Dict) -> Tuple:
        """Generate output from Medgemma model"""
        import torch
        
        self.medgemma_model.eval()
        
        with torch.inference_mode():
            try:
                outputs = self.medgemma_model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True,
                    pad_token_id=self.medgemma_processor.tokenizer.pad_token_id or self.medgemma_processor.tokenizer.eos_token_id
                )
                return outputs, inputs["input_ids"].shape[-1]
            except Exception as e:
                logger.error(f"Generation failed: {e}", exc_info=True)
                raise
            finally:
                # Clear input tensors from GPU memory immediately after generation
                try:
                    if torch.cuda.is_available():
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor) and value.is_cuda:
                                del value
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"Error clearing input tensors: {e}")
    
    def _extract_answer_from_output(self, outputs, input_len: int, query: str, inputs: Dict) -> Optional[str]:
        """Extract and clean answer from model output"""
        output_length = outputs[0].shape[0]
        
        if output_length <= input_len:
            logger.error(f"No new tokens generated (input: {input_len}, output: {output_length})")
            return None
        
        generated_tokens = outputs[0][input_len:]
        if len(generated_tokens) == 0:
            logger.warning("No new tokens in output")
            return None
        
        answer = self.medgemma_processor.decode(generated_tokens, skip_special_tokens=True)
        full_output = self.medgemma_processor.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"Generated text preview: {answer[:200]}")
        
        # Clean template markers
        answer = self._clean_template_markers(answer, query, full_output, inputs)
        
        # Validate answer length
        if not answer or len(answer.strip()) < 10:
            answer = self._try_alternative_extraction(full_output, inputs, answer)
            if not answer or len(answer.strip()) < 10:
                logger.error("All extraction methods failed")
                return None
        
        return answer
    
    def _clean_template_markers(self, answer: str, query: str, full_output: str, inputs: Dict) -> str:
        """Remove chat template markers and clean up answer"""
        if not answer:
            return answer
        
        # Pattern 1: <start_of_turn>model\nAnswer<end_of_turn>
        if "<start_of_turn>model" in answer:
            parts = answer.split("<start_of_turn>model", 1)
            if len(parts) > 1:
                model_part = parts[1]
                if "<end_of_turn>" in model_part:
                    model_part = model_part.split("<end_of_turn>")[0]
                answer = model_part.strip()
        
        # Pattern 2: Remove "model" or "assistant" prefixes
        if answer.strip().startswith("model"):
            answer = answer.replace("model", "", 1).strip()
        if answer.strip().startswith("assistant"):
            answer = answer.replace("assistant", "", 1).strip()
        
        # Remove common markers
        markers_to_remove = ["<start_of_turn>", "<end_of_turn>", "model", "user", "assistant"]
        for marker in markers_to_remove:
            answer = answer.replace(marker, "").strip()
        
        answer = answer.strip()
        
        # Remove leading separators
        while answer and answer[0] in [":", "-", ".", " "]:
            answer = answer[1:].strip()
        
        # Remove question if repeated at start
        if query.lower() in answer.lower()[:len(query)+50]:
            query_lower = query.lower()
            answer_lower = answer.lower()
            query_pos = answer_lower.find(query_lower)
            if query_pos >= 0:
                after_query = answer[query_pos + len(query):]
                for prefix in [":", ".", "-", "is", "are", "means", "refers to"]:
                    if after_query.strip().lower().startswith(prefix):
                        after_query = after_query.strip()[len(prefix):].strip()
                if len(after_query.strip()) > 10:
                    answer = after_query.strip()
        
        # Final cleanup
        answer = answer.strip()
        query_lower = query.lower().strip()
        answer_lower = answer.lower().strip()
        if answer_lower.startswith(query_lower):
            remaining = answer[len(query):].strip()
            for sep in [":", ".", "-", "is", "are", "means"]:
                if remaining.lower().startswith(sep):
                    remaining = remaining[len(sep):].strip()
            if len(remaining) > 10:
                answer = remaining
        
        answer = answer.replace("<|im_start|>", "").replace("<|im_end|>", "")
        answer = answer.replace("<|endoftext|>", "")
        return answer.strip()
    
    def _try_alternative_extraction(self, full_output: str, inputs: Dict, current_answer: Optional[str]) -> Optional[str]:
        """Try alternative methods to extract answer from full output"""
        if current_answer and len(current_answer.strip()) >= 10:
            return current_answer
        
        # Method 1: Look for model's response after <start_of_turn>model
        if "<start_of_turn>model" in full_output:
            parts = full_output.split("<start_of_turn>model", 1)
            if len(parts) > 1:
                model_response = parts[1]
                if "<end_of_turn>" in model_response:
                    model_response = model_response.split("<end_of_turn>")[0]
                model_response = model_response.strip()
                if len(model_response) > 10:
                    return model_response
        
        # Method 2: Extract after input prompt
        if hasattr(self.medgemma_processor, 'apply_chat_template'):
            input_text = self.medgemma_processor.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if input_text in full_output:
                answer_part = full_output[full_output.find(input_text) + len(input_text):]
                answer_part = answer_part.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
                answer_part = answer_part.replace("model", "").replace("user", "").strip()
                if len(answer_part) > 10:
                    return answer_part
        
        # Method 3: Extract everything after input
        if hasattr(self.medgemma_processor, 'apply_chat_template'):
            try:
                input_decoded = self.medgemma_processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
                if input_decoded in full_output:
                    potential_answer = full_output.split(input_decoded, 1)[-1]
                    potential_answer = potential_answer.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
                    potential_answer = potential_answer.replace("model", "").replace("user", "").replace("assistant", "").strip()
                    if len(potential_answer) > 10:
                        logger.debug(f"Answer extracted via alternative method")
                        return potential_answer
            except Exception as e:
                logger.warning(f"Alternative extraction failed: {e}")
        
        return current_answer
    
    def _remove_duplicate_leading_content(self, text: str) -> str:
        """Remove repeated intro/prefix so the same phrase does not appear twice (e.g. truncated then full)."""
        if not text or len(text.strip()) < 20:
            return text
        # Split by double newlines (paragraphs)
        parts = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        if len(parts) >= 2:
            first = parts[0]
            second = parts[1]
            # If first part is a prefix of the second, keep only the longer/full version
            if len(first) >= 15 and (second.startswith(first) or second.startswith(first[:min(len(first), 100)])):
                return '\n\n'.join(parts[1:])
            if len(first) >= 15 and first.lower() in second.lower()[: len(second)][: len(first) + 80]:
                return '\n\n'.join(parts[1:])
        # Single newline: same intro then full (e.g. "short intro\nfull intro and rest")
        if '\n' in text and text.count('\n') == 1:
            a, b = text.split('\n', 1)
            a, b = a.strip(), b.strip()
            if len(a) >= 15 and (b.startswith(a) or a in b[: len(a) + 80]):
                return b
        # Single paragraph: first sentence repeated at start of rest
        first_sentence_end = max(
            text.find('. ', 20) + 1 if '. ' in text[20:80] else 0,
            text.find('。', 20) + 1 if '。' in text[20:80] else 0,
        )
        if first_sentence_end > 30:
            prefix = text[:first_sentence_end].strip()
            rest = text[first_sentence_end:].strip()
            if rest and len(prefix) >= 20 and (rest.startswith(prefix[:50]) or prefix[:40] in rest[:120]):
                return text[first_sentence_end:].strip()
        return text

    def _format_medgemma_response(self, answer: str) -> str:
        """Format Medgemma response for better readability"""
        if not answer:
            return answer
        
        # Remove duplicated leading content (e.g. same intro repeated when AI insight is selected)
        answer = self._remove_duplicate_leading_content(answer)
        
        # Remove markdown formatting
        answer = re.sub(r'\*\s*\*\s*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
        answer = re.sub(r'\*', '', answer)
        
        # Normalize whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        # Format headings
        answer = self._format_headings(answer)
        
        # Fix punctuation spacing
        answer = re.sub(r'\s+([.,!?;:।])', r'\1', answer)
        answer = re.sub(r'([.,!?;:।])([^\s])', r'\1 \2', answer)
        answer = re.sub(r'([.,!?;:।]){2,}', r'\1', answer)
        
        # Format paragraphs
        answer = self._format_paragraphs(answer)
        
        # Remove markdown artifacts
        answer = re.sub(r'#{1,6}\s*', '', answer)
        answer = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', answer)
        answer = re.sub(r'`([^`]+)`', r'\1', answer)
        
        # Truncate to word limit
        answer = self._truncate_to_word_limit(answer, max_words=150)
        
        # Ensure complete sentences and proper punctuation
        answer = self._ensure_punctuation_and_complete_sentences(answer)
        
        # Final cleanup
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
                cleaned_lines.append(cleaned_line)
            else:
                cleaned_lines.append('')
        answer = '\n'.join(cleaned_lines)
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        return answer.strip()
    
    def _format_headings(self, text: str) -> str:
        """Detect and format headings in text"""
        heading_patterns = [
            (r'([A-Z\u0900-\u097F][^।.!?\n]{2,50}?):\s+([A-Z\u0900-\u097F][^।.!?\n]{10,})', 
             lambda m: f'\n\n{m.group(1)}:\n{m.group(2)}'),
            (r'([A-Z\u0900-\u097F][^।.!?\n]{2,50}?)\s*-\s+([A-Z\u0900-\u097F][^।.!?\n]{10,})',
             lambda m: f'\n\n{m.group(1)}:\n{m.group(2)}'),
        ]
        
        for pattern, replacement in heading_patterns:
            def format_heading(match):
                heading_text = match.group(1).strip()
                word_count = len(heading_text.split())
                if 2 <= word_count <= 8 and len(heading_text) < 60:
                    return replacement(match)
                return match.group(0)
            text = re.sub(pattern, format_heading, text)
        
        # Additional pass for standalone headings
        def enhance_headings(text):
            parts = re.split(r'([।.!?]\s+)', text)
            result = []
            for i, part in enumerate(parts):
                heading_match = re.search(r'([A-Z\u0900-\u097F][^:।.!?\n]{2,50}?):\s*$', part)
                if heading_match and i + 1 < len(parts):
                    heading_text = heading_match.group(1).strip()
                    word_count = len(heading_text.split())
                    if 2 <= word_count <= 8:
                        result.append(f'\n\n{heading_text}:\n')
                        continue
                result.append(part)
            return ''.join(result)
        
        return enhance_headings(text)
    
    def _format_paragraphs(self, text: str) -> str:
        """Format text into well-structured paragraphs"""
        parts = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        
        if len(parts) > 1:
            paragraphs = []
            i = 0
            while i < len(parts):
                part = parts[i]
                if part.endswith(':') and len(part.split()) <= 8:
                    heading = part
                    if i + 1 < len(parts):
                        explanation = parts[i + 1]
                        if explanation and explanation.strip() and len(explanation.split()) >= 5:
                            paragraphs.append(f'{heading}\n{explanation}')
                            i += 2
                        else:
                            i += 2
                    else:
                        i += 1
                else:
                    paragraphs.append(part)
                    i += 1
        else:
            heading_matches = list(re.finditer(r'([^।.!?\n]{5,60}):\s+', text))
            if heading_matches:
                paragraphs = []
                last_end = 0
                for match in heading_matches:
                    before = text[last_end:match.start()].strip()
                    if before:
                        paragraphs.append(before)
                    
                    heading = match.group(1).strip()
                    heading_end = match.end()
                    next_match = heading_matches[heading_matches.index(match) + 1] if heading_matches.index(match) + 1 < len(heading_matches) else None
                    if next_match:
                        explanation = text[heading_end:next_match.start()].strip()
                    else:
                        explanation = text[heading_end:].strip()
                    
                    if explanation and explanation.strip() and len(explanation.split()) >= 5:
                        if len(heading.split()) <= 8:
                            paragraphs.append(f'{heading}: {explanation}')
                        else:
                            paragraphs.append(before + ' ' + heading + ': ' + explanation)
                    
                    last_end = next_match.start() if next_match else len(text)
            else:
                sentences = re.split(r'([.!?।]\s+)', text)
                paragraphs = []
                current_para = []
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        current_para.append(sentence)
                        if len(current_para) >= 3 or i == len(sentences) - 1:
                            para_text = ''.join(current_para).strip()
                            if para_text:
                                paragraphs.append(para_text)
                            current_para = []
                if current_para:
                    para_text = ''.join(current_para).strip()
                    if para_text:
                        paragraphs.append(para_text)
        
        # Format paragraphs
        formatted_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if '\n' in para and para.count('\n') == 1:
                lines = para.split('\n', 1)
                heading = lines[0].strip()
                explanation = lines[1].strip() if len(lines) > 1 else ''
                
                if not explanation or not explanation.strip() or len(explanation.split()) < 5:
                    continue
                
                formatted_heading = heading
                if not formatted_heading.endswith(':'):
                    formatted_heading = formatted_heading.rstrip(':') + ':'
                
                if explanation and explanation[0].islower():
                    explanation = explanation[0].upper() + explanation[1:]
                explanation = re.sub(r'\s+', ' ', explanation).strip()
                formatted_paragraphs.append(f'{formatted_heading}\n{explanation}')
            else:
                sentences = re.split(r'([.!?।]\s+)', para)
                formatted_sentences = []
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        if i == 0 or (i > 0 and sentences[i-1].strip().endswith(('.', '!', '?', '।'))):
                            sentence = sentence.strip()
                            if sentence and sentence[0].islower() and not sentence[0].isdigit():
                                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                        formatted_sentences.append(sentence)
                para = ''.join(formatted_sentences)
                para = re.sub(r'\s+', ' ', para).strip()
                if para:
                    formatted_paragraphs.append(para)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _truncate_to_word_limit(self, text: str, max_words: int = 150) -> str:
        """Truncate text to word limit while preserving structure"""
        word_count = len(re.findall(r'\b\w+\b', text))
        if word_count <= max_words:
            logger.debug(f"Response: {word_count} words (within limit)")
            return text
        
        paragraphs = text.split('\n\n')
        truncated_paragraphs = []
        current_word_count = 0
        
        for para in paragraphs:
            para_words = len(re.findall(r'\b\w+\b', para))
            
            if current_word_count + para_words <= max_words:
                truncated_paragraphs.append(para)
                current_word_count += para_words
            else:
                remaining_words = max_words - current_word_count
                if remaining_words > 10:
                    if '\n' in para:
                        lines = para.split('\n', 1)
                        heading = lines[0]
                        explanation = lines[1] if len(lines) > 1 else ''
                        
                        if not explanation or not explanation.strip():
                            break
                        
                        expl_words = len(re.findall(r'\b\w+\b', explanation))
                        
                        if expl_words <= remaining_words and remaining_words >= 15:
                            words = explanation.split()
                            truncated_expl = ' '.join(words[:remaining_words])
                            last_sent = max(
                                truncated_expl.rfind('.'),
                                truncated_expl.rfind('!'),
                                truncated_expl.rfind('?'),
                                truncated_expl.rfind('।')
                            )
                            if last_sent > len(truncated_expl) * 0.7:
                                truncated_expl = truncated_expl[:last_sent + 1]
                            else:
                                truncated_expl = truncated_expl.rstrip() + '...'
                            truncated_paragraphs.append(f'{heading}\n{truncated_expl}')
                        else:
                            break
                    else:
                        words = para.split()
                        truncated_para = ' '.join(words[:remaining_words])
                        last_sent = max(
                            truncated_para.rfind('.'),
                            truncated_para.rfind('!'),
                            truncated_para.rfind('?'),
                            truncated_para.rfind('।')
                        )
                        if last_sent > len(truncated_para) * 0.7:
                            truncated_para = truncated_para[:last_sent + 1]
                        else:
                            truncated_para = truncated_para.rstrip() + '...'
                        truncated_paragraphs.append(truncated_para)
                break
        
        result = '\n\n'.join(truncated_paragraphs)
        final_word_count = len(re.findall(r'\b\w+\b', result))
        logger.debug(f"Response truncated: {word_count} → {final_word_count} words")
        return result
    
    def _ensure_punctuation_and_complete_sentences(self, text: str) -> str:
        """Ensure text ends with proper punctuation and fix incomplete last sentence."""
        if not text or not text.strip():
            return text
        text = text.strip()
        # Ensure single space before sentence-ending punctuation
        text = re.sub(r'\s+([.!?।])', r'\1', text)
        text = re.sub(r'([.!?।])([^\s.!?।])', r'\1 \2', text)
        # If text ends mid-sentence (no . ! ? । at end), add ellipsis or period
        if text and text[-1] not in '.!?।':
            # If last word looks truncated (very short or incomplete), use ellipsis
            last_word = (text.split() or [''])[-1]
            if len(last_word) <= 2 or last_word.lower() in ('a', 'an', 'the', 'is', 'to', 'for'):
                text = text.rstrip() + '...'
            else:
                text = text.rstrip() + '.'
        return text
    
    def _cache_medgemma_response(self, query: str, language: str, context: Optional[str], answer: str):
        """Cache Medgemma response"""
        if not answer or not self.redis_client:
            return
        
        cache_key = self._get_cache_key(query, language, f"medgemma_{hash(context or '')}")
        cache_ttl = getattr(settings, 'CACHE_TTL_MEDGEMMA', 86400)
        try:
            self.redis_client.setex(cache_key, cache_ttl, answer.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def get_answer_from_medgemma(
        self,
        query: str,
        language: str = "en",
        context: Optional[str] = None,
        use_cache: bool = True,
        image: Optional[Union[str, bytes, Any]] = None,
    ) -> Optional[str]:
        """
        Get answer from MedGemma multimodal model (text-only or image + text).

        When `image` is provided (base64 string, bytes, or PIL Image), the model
        receives multimodal input and cache is skipped.

        GPU memory is cleared before inference and released after each response.

        Args:
            query: User's question
            language: Language code ('en' or 'hi')
            context: Optional context from conversation history
            use_cache: Whether to use cached responses (ignored when image is provided)
            image: Optional image for multimodal: base64 string, bytes, or PIL Image

        Returns:
            Generated answer or None if model not available
        """
        pil_image = None
        if image is not None:
            pil_image = _decode_image_for_medgemma(image)
            if pil_image is None:
                logger.warning("Image provided but decode failed; falling back to text-only")
            else:
                use_cache = False

        # Check cache first (text-only)
        if use_cache:
            cached_answer = self._get_medgemma_cached_answer(query, language, context)
            if cached_answer:
                return cached_answer
        
        # CRITICAL: Clear GPU memory BEFORE inference to ensure clean state
        # This prevents memory accumulation from previous requests
        self._clear_gpu_memory_before_inference()
        
        # CRITICAL: Check if model manager has leftover model from previous run
        # This can happen if the process wasn't fully killed and singleton persists
        try:
            if (hasattr(self._model_manager, 'medgemma_model') and 
                self._model_manager.medgemma_model is not None and
                not self._model_manager.medgemma_loaded):
                logger.warning("Detected leftover model in manager (inconsistent state), cleaning up...")
                self._model_manager.cleanup(force=True)
        except Exception as e:
            logger.debug(f"Error checking for leftover model: {e}")
        
        # Lazy load Medgemma if not already loaded
        self._ensure_medgemma_loaded()
        
        if not self.medgemma_model or not self.medgemma_processor:
            logger.warning("Medgemma not available")
            return None
        
        # Log GPU memory after model is loaded (before inference)
        self._log_gpu_memory_stats("AFTER MODEL LOAD")
        
        logger.info(f"Generating Medgemma response: {query[:50]}")
        
        inputs = None
        outputs = None
        answer = None
        
        try:
            # Prepare prompt
            question_text, system_prompt = self._prepare_medgemma_prompt(query, context)

            # Prepare inputs (text-only or image + text for multimodal)
            inputs, input_len = self._prepare_medgemma_inputs(
                question_text, system_prompt, image=pil_image
            )

            # Generate output
            outputs, input_len = self._generate_medgemma_output(inputs)
            
            # Extract answer
            answer = self._extract_answer_from_output(outputs, input_len, query, inputs)
            
            # Log GPU memory after inference (before cleanup)
            self._log_gpu_memory_stats("AFTER INFERENCE")
            
            # Process answer if available
            if answer:
                logger.debug(f"Medgemma answer ready ({len(answer)} chars)")
                
                # Translate to Hindi if needed
                if language == 'hi':
                    try:
                        answer = translation_service.translate_to_hindi(answer)
                    except:
                        logger.warning("Translation to Hindi failed")
                
                # Format response
                answer = self._format_medgemma_response(answer)
                
                # Cache response
                if use_cache:
                    self._cache_medgemma_response(query, language, context, answer)
            else:
                logger.warning("Medgemma generated no answer")
            
            return answer
            
        except Exception as e:
            logger.error(f"Medgemma generation failed: {e}", exc_info=True)
            return None
            
        finally:
            # CRITICAL: ALWAYS release GPU memory after each MedGemma response
            # This ensures GPU is freed after every bot response, even on errors
            # Use try-finally to guarantee cleanup happens
            try:
                # Clear inference tensors from GPU memory
                import torch
                if torch.cuda.is_available():
                    if inputs:
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor) and value.is_cuda:
                                del value
                    if outputs is not None:
                        if isinstance(outputs, torch.Tensor) and outputs.is_cuda:
                            del outputs
                        elif isinstance(outputs, (list, tuple)):
                            for item in outputs:
                                if isinstance(item, torch.Tensor) and item.is_cuda:
                                    del item
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"Error clearing inference tensors: {e}")
            
            # Always call the main cleanup method
            try:
                self._release_gpu_memory_after_inference()
            except Exception as e:
                logger.error(f"CRITICAL: Cleanup failed in finally block: {e}", exc_info=True)
                # Last resort cleanup
                try:
                    import torch
                    import gc
                    # Force cleanup of model manager
                    if hasattr(self, '_model_manager'):
                        try:
                            if hasattr(self._model_manager, 'medgemma_model') and self._model_manager.medgemma_model:
                                logger.error("EMERGENCY: Model still exists, forcing deletion...")
                                model = self._model_manager.medgemma_model
                                if hasattr(model, 'to'):
                                    try:
                                        model = model.to('cpu')
                                    except:
                                        pass
                                del model
                                self._model_manager.medgemma_model = None
                                self._model_manager.medgemma_processor = None
                                self._model_manager.medgemma_tokenizer = None
                                self._model_manager.medgemma_loaded = False
                        except:
                            pass
                    
                    # Aggressive cleanup
                    for _ in range(10):
                        gc.collect()
                    
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            with torch.cuda.device(i):
                                for _ in range(10):
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()
                                torch.cuda.synchronize()
                    
                    logger.warning("Emergency cleanup completed")
                except Exception as e2:
                    logger.error(f"Even emergency cleanup failed: {e2}")
            
            # Final verification - log memory state
            try:
                final_stats = self._get_gpu_memory_stats()
                if final_stats:
                    for dev_id, mem_info in final_stats.items():
                        if mem_info['allocated_gb'] > 2.0:
                            logger.error(
                                f"CRITICAL: GPU {dev_id} memory still high after all cleanup attempts: "
                                f"{mem_info['allocated_gb']:.2f}GB allocated, {mem_info['reserved_gb']:.2f}GB reserved"
                            )
            except:
                pass