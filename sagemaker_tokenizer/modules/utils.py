"""
Utility functions for memory management and optimization
"""
import gc
import logging
import torch

LOGGER = logging.getLogger("tokenizer")

def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory"""
    try:
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            
            # Conservative batch sizing based on available memory
            if free_memory_gb > 10:
                batch_size = 32
            elif free_memory_gb > 6:
                batch_size = 16
            elif free_memory_gb > 3:
                batch_size = 8
            else:
                batch_size = 4
                
            LOGGER.info("GPU: %.1f GB total, %.1f GB free → batch_size=%d", total_memory_gb, free_memory_gb, batch_size)
            return batch_size
        else:
            return 8  # CPU fallback
    except Exception as e:
        LOGGER.warning("Could not determine optimal batch size: %s, using 8", e)
        return 8  # Safe fallback


def cleanup_memory():
    """Force garbage collection and clear CUDA cache if available"""
    gc.collect()
    try:
        if torch.cuda.is_available():
            # More aggressive CUDA cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Clean up inter-process communication
            # Force synchronization to ensure cleanup is complete
            torch.cuda.synchronize()
            LOGGER.info("✓ CUDA cache cleared aggressively")
            
            # Log memory status after cleanup
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            LOGGER.info("GPU memory: %.2f GB allocated, %.2f GB reserved", memory_allocated, memory_reserved)
    except ImportError:
        pass
    LOGGER.info("✓ Memory cleanup completed")
