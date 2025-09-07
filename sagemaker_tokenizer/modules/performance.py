"""
Performance tracking utilities for monitoring processing metrics
"""
import logging
import time
import psutil
from datetime import datetime
from typing import Dict, Any

LOGGER = logging.getLogger("tokenizer")

class PerformanceTracker:
    """Track performance metrics during processing"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_chunks = 0
        self.files_processed = 0
        self.files_skipped = 0
        self.memory_samples = []
        self.process = psutil.Process()
        
    def add_file_stats(self, tokens: int, chunks: int, skipped: bool = False):
        """Add statistics from processing a file"""
        # Convert numpy types to native Python types
        self.total_tokens += int(tokens)
        self.total_chunks += int(chunks)
        
        if skipped:
            self.files_skipped += 1
        else:
            self.files_processed += 1
        
        # Sample memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return performance metrics"""
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        
        metrics = {
            "processing_time": {
                "total_seconds": round(elapsed_time, 2),
                "total_minutes": round(elapsed_minutes, 2)
            },
            "throughput": {
                "tokens_per_minute": round(self.total_tokens / elapsed_minutes) if elapsed_minutes > 0 else 0,
                "chunks_per_minute": round(self.total_chunks / elapsed_minutes) if elapsed_minutes > 0 else 0,
                "files_per_minute": round(self.files_processed / elapsed_minutes) if elapsed_minutes > 0 else 0
            },
            "totals": {
                "total_tokens_processed": int(self.total_tokens),
                "total_chunks_processed": int(self.total_chunks),
                "files_processed": int(self.files_processed),
                "files_skipped": int(self.files_skipped),
                "total_files_seen": int(self.files_processed + self.files_skipped)
            },
            "memory_usage": {
                "peak_memory_mb": round(max(self.memory_samples)) if self.memory_samples else 0,
                "avg_memory_mb": round(sum(self.memory_samples) / len(self.memory_samples)) if self.memory_samples else 0,
                "memory_per_chunk_mb": round(sum(self.memory_samples) / len(self.memory_samples) / self.total_chunks, 4) if self.memory_samples and self.total_chunks > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
