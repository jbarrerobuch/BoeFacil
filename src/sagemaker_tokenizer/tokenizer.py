"""
SageMaker Training Job script to tokenize text chunks stored in parquet files.

Behavior:
- Reads all .parquet files from --input-path (non-recursive by default).
- Tokenizes the `texto` column (configurable) using a HuggingFace tokenizer.
- Writes tokenized parquet files to --output-path keeping the same filename.
- Emits a `results.json` with summary statistics.

Designed to run inside a SageMaker Training Job with spot instances where input is mounted at
`/opt/ml/input/data/dataset` and output at `/opt/ml/output/data`.

SageMaker Job will save checkpoints to handle spot instance interruptions.

img: 763104351884.dkr.ecr.eu-west-3.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04

"""

import argparse
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import time
import psutil
import atexit
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Set, Optional
import pandas as pd

def install_requirements():
    """Install required packages if not available"""
    print("Installing packages for HuggingFace PyTorch Inference 2.6.0...")
    
    # Configure CUDA memory management BEFORE any PyTorch operations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✓ CUDA memory fragmentation prevention enabled")
    
    # This image already has PyTorch 2.6.0 and Transformers 4.49.0 pre-installed
    print("✓ PyTorch 2.6.0 GPU and Transformers 4.49.0 already available")
    
    # Install sentence-transformers compatible with existing numpy 1.26.4
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "sentence-transformers>=3.3.0", "numpy<2.0", "psutil"
    ])
    print("✓ sentence_transformers and psutil installed with compatible numpy")
    
    # Don't upgrade pandas/pyarrow - use what's already installed
    print("✓ Using pre-installed pandas and pyarrow to avoid conflicts")

# Install dependencies BEFORE importing them
print("=== Installing dependencies ===")
install_requirements()
print("=== Dependencies installed ===")

try:
    from sentence_transformers import SentenceTransformer
    SENTEVAL_AVAILABLE = True
    print("✓ sentence_transformers imported successfully")
except Exception as e:
    print(f"✗ Failed to import sentence_transformers after installation: {e}")
    SentenceTransformer = None
    SENTEVAL_AVAILABLE = False

LOGGER = logging.getLogger("tokenizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class CheckpointManager:
    """Manage checkpointing for spot instance resilience in SageMaker Training Jobs"""
    
    def __init__(self, checkpoint_dir: Path, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Ensure checkpoint directory exists
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self.processed_files: Set[str] = set()
        self.interrupted = False
        
        # Register signal handlers for spot termination
        signal.signal(signal.SIGTERM, self._handle_termination)
        signal.signal(signal.SIGINT, self._handle_termination)
        # Register exit handler
        atexit.register(self._save_checkpoint)
        
        # Try to load previous checkpoint
        self._load_checkpoint()
    
    def _handle_termination(self, signum, frame):
        """Handle termination signals"""
        LOGGER.warning("Received termination signal %d - saving checkpoint", signum)
        self.interrupted = True
        self._save_checkpoint()
        # Exit gracefully after saving
        sys.exit(0)
    
    def _save_checkpoint(self):
        """Save checkpoint to disk using SageMaker Training Jobs checkpoint directory"""
        # Ensure both directories exist
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "processed_files": list(self.processed_files),
            "timestamp": datetime.now().isoformat(),
            "interrupted": self.interrupted
        }
        
        try:
            with self.checkpoint_file.open("w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            if self.interrupted:
                LOGGER.info("✓ Saved checkpoint before termination: %s", self.checkpoint_file)
            else:
                LOGGER.debug("✓ Updated checkpoint: %s", self.checkpoint_file)
                
        except Exception as e:
            LOGGER.error("Failed to save checkpoint: %s", e)
    
    def _load_checkpoint(self):
        """Load checkpoint from disk"""
        if not self.checkpoint_file.exists():
            LOGGER.info("No previous checkpoint found, starting fresh")
            return
        
        try:
            with self.checkpoint_file.open("r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            self.processed_files = set(checkpoint_data.get("processed_files", []))
            LOGGER.info("✓ Loaded checkpoint with %d previously processed files", len(self.processed_files))
            
        except Exception as e:
            LOGGER.error("Failed to load checkpoint: %s", e)
    
    def mark_file_processed(self, filename: str):
        """Mark a file as successfully processed"""
        self.processed_files.add(filename)
        # Update checkpoint file periodically
        if len(self.processed_files) % 5 == 0:  # Every 5 files
            self._save_checkpoint()
    
    def is_file_processed(self, filename: str) -> bool:
        """Check if a file was already processed"""
        return filename in self.processed_files
    
    def get_processed_files(self) -> Set[str]:
        """Get set of processed files"""
        return self.processed_files


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


def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            
            # Conservative batch sizing based on available memory
            if free_memory_gb > 10:  # Lots of free memory
                batch_size = 32
            elif free_memory_gb > 6:  # Moderate memory
                batch_size = 16  
            elif free_memory_gb > 3:  # Limited memory
                batch_size = 8
            else:  # Very limited memory
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
        import torch
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


def iter_parquet_files(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".parquet":
            yield p


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def embed_texts(texts: List[str], model: Any, batch_size: int) -> List[List[float]]:
    """Return list of embeddings (list of floats) for the input texts.

    Uses SentenceTransformer.encode under the hood with the configured batch_size.
    Includes retry logic for CUDA memory errors.
    """
    if model is None:
        raise RuntimeError("sentence_transformers not available; cannot produce embeddings")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # encode returns a numpy array; convert to python lists for parquet-friendly storage
            embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and attempt < max_retries - 1:
                LOGGER.warning("CUDA OOM on attempt %d, reducing batch size and retrying...", attempt + 1)
                cleanup_memory()
                # Reduce batch size for retry
                batch_size = max(1, batch_size // 2)
                LOGGER.info("Reduced batch size to %d for retry", batch_size)
            else:
                raise e


def process_file(
    input_path: Path,
    output_path: Path,
    tokenizer,
    text_column: str,
    batch_size: int,
    max_length: int,
    tracker: PerformanceTracker,
):
    LOGGER.info("Reading parquet: %s", input_path)
    df = pd.read_parquet(input_path)
    LOGGER.info("Loaded %d rows from %s", len(df), input_path)

    if text_column not in df.columns:
        LOGGER.warning("Column '%s' not found in %s; adding empty strings", text_column, input_path)
        df[text_column] = ""

    # Extract token count if available
    tokens_in_file = 0
    if "tokens_aproximados" in df.columns:
        tokens_in_file = int(df["tokens_aproximados"].sum())  # Convert numpy.int64 to int
        LOGGER.info("File contains %d approximate tokens", tokens_in_file)
    
    chunks_in_file = len(df)
    LOGGER.info("Processing %d chunks in this file", chunks_in_file)

    texts = df[text_column].fillna("").astype(str).tolist()

    all_embeddings: List[List[float]] = []
    batch_count = 0
    for batch in batch_iter(texts, batch_size):
        emb = embed_texts(batch, model=tokenizer, batch_size=batch_size)
        all_embeddings.extend(emb)
        batch_count += 1
        
        # Clean memory more frequently to prevent accumulation
        if batch_count % 3 == 0:  # Every 3 batches instead of 10
            cleanup_memory()
            LOGGER.info("Memory cleanup after batch %d", batch_count)

    # Attach embeddings to DataFrame
    df["embeddings"] = all_embeddings

    # Clean intermediate data to free memory
    del texts, all_embeddings
    cleanup_memory()

    # Write out to output_path with same name
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / input_path.name
    LOGGER.info("Writing tokenized parquet: %s", out_file)
    df.to_parquet(out_file, index=False)
    
    # Verify file was written
    if out_file.exists():
        file_size = out_file.stat().st_size
        LOGGER.info("✓ Successfully wrote %s (%d bytes)", out_file, file_size)
    else:
        LOGGER.error("✗ Failed to write %s - file does not exist", out_file)

    # Update performance tracker
    tracker.add_file_stats(tokens_in_file, chunks_in_file)

    # Final cleanup after processing each file
    del df
    cleanup_memory()
    
    return tokens_in_file, chunks_in_file


def main():
    parser = argparse.ArgumentParser(description="Produce embeddings for parquet text files (SentenceTransformers)")
    parser.add_argument("--input-path", type=str, default="/opt/ml/input/data/dataset", 
                       help="Path to input data directory (default: SageMaker Training Job dataset channel)")
    parser.add_argument("--output-path", type=str, default="/opt/ml/output/data", 
                       help="Path to output directory (default: SageMaker Training Job output directory)")
    parser.add_argument("--checkpoint-path", type=str, default="/opt/ml/checkpoints",
                       help="Path to checkpoints directory (default: SageMaker Training Job checkpoints directory)")
    parser.add_argument("--model", type=str, default="pablosi/bge-m3-trained-2", help="SentenceTransformers model id")
    parser.add_argument("--text-column", type=str, default="texto", help="Name of the text column to embed")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = auto-detect based on GPU)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing (process all files)")
    args = parser.parse_args()

    # Auto-detect optimal batch size if not specified
    if args.batch_size == 0:
        args.batch_size = get_optimal_batch_size()
        LOGGER.info("Auto-detected optimal batch size: %d", args.batch_size)

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    checkpoint_dir = Path(args.checkpoint_path)
    
    # Initialize checkpoint manager for spot instance resilience in Training Jobs
    checkpoint_mgr = CheckpointManager(checkpoint_dir, output_dir)

    # Debug: Show what we have in the training environment
    LOGGER.info("=== DEBUG INFO ===")
    LOGGER.info("Input path requested: %s", input_dir)
    LOGGER.info("Output path requested: %s", output_dir)
    LOGGER.info("Input path exists: %s", input_dir.exists())
    
    # List contents of input data directories used in Training Jobs
    training_input = Path("/opt/ml/input/data")
    if training_input.exists():
        LOGGER.info("Contents of /opt/ml/input/data:")
        for item in training_input.iterdir():
            LOGGER.info("  - %s (type: %s)", item, "dir" if item.is_dir() else "file")
            if item.is_dir():
                try:
                    for subitem in item.iterdir():
                        LOGGER.info("    - %s", subitem)
                except Exception as e:
                    LOGGER.info("    - Error listing contents: %s", e)
    else:
        LOGGER.info("/opt/ml/input/data does not exist")
        
    # Also check if dataset channel exists specifically
    dataset_dir = Path("/opt/ml/input/data/dataset")
    if dataset_dir.exists():
        LOGGER.info("Dataset channel directory exists")
    else:
        LOGGER.info("Dataset channel directory does not exist")
    
    LOGGER.info("=== END DEBUG ===")

    if not input_dir.exists():
        raise SystemExit(f"Input path does not exist: {input_dir}")

    # Initialize performance tracker
    tracker = PerformanceTracker()
    LOGGER.info("Starting processing with performance tracking and checkpointing")

    # Load tokenizer if available
    tokenizer = None
    if SENTEVAL_AVAILABLE:
        try:
            model = SentenceTransformer(args.model)
            tokenizer = model
            LOGGER.info("Loaded SentenceTransformer model: %s", args.model)
        except Exception as e:
            LOGGER.exception("Failed to load SentenceTransformer model '%s': %s", args.model, e)
            raise SystemExit(2)
    else:
        LOGGER.error("sentence_transformers package not available; cannot produce embeddings")
        raise SystemExit(2)

    processed = 0
    files = []
    total_tokens = 0
    total_chunks = 0
    
    # Report checkpoint status
    already_processed = len(checkpoint_mgr.get_processed_files())
    if already_processed > 0 and not args.no_checkpoint:
        LOGGER.info("Resuming from checkpoint: %d files already processed", already_processed)

    # Process all parquet files
    parquet_files = list(iter_parquet_files(input_dir))
    LOGGER.info("Found %d parquet files to process", len(parquet_files))
    
    for p in parquet_files:
        # Check if this file was already processed in a previous run
        if not args.no_checkpoint and checkpoint_mgr.is_file_processed(p.name):
            LOGGER.info("Skipping %s (already processed in previous run)", p.name)
            
            # Estimate file stats for reporting
            try:
                df = pd.read_parquet(p)
                tokens_in_file = 0
                if "tokens_aproximados" in df.columns:
                    tokens_in_file = int(df["tokens_aproximados"].sum())
                chunks_in_file = len(df)
                tracker.add_file_stats(tokens_in_file, chunks_in_file, skipped=True)
                LOGGER.info("Skipped file %s: %d tokens, %d chunks", p.name, tokens_in_file, chunks_in_file)
            except Exception:
                # If we can't read the file, just skip without reporting stats
                tracker.add_file_stats(0, 0, skipped=True)
                
            continue

        try:
            tokens, chunks = process_file(p, output_dir, tokenizer, args.text_column, args.batch_size, 1024, tracker)
            processed += 1
            files.append(p.name)
            total_tokens += tokens
            total_chunks += chunks
            LOGGER.info("File %s: %d tokens, %d chunks", p.name, tokens, chunks)
            
            # Mark file as processed in the checkpoint
            checkpoint_mgr.mark_file_processed(p.name)
            
        except Exception as e:
            LOGGER.exception("Failed to process %s: %s", p, e)

    # Get performance metrics
    metrics = tracker.get_metrics()
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{timestamp}.json"

    results = {
        "status": "completed", 
        "files_processed": processed, 
        "files": files,
        "performance_metrics": metrics,
        "checkpoint_info": {
            "files_in_checkpoint": len(checkpoint_mgr.get_processed_files()),
            "checkpoint_location": str(checkpoint_mgr.checkpoint_file),
            "checkpointing_enabled": not args.no_checkpoint,
            "files_skipped": metrics['totals']['files_skipped']
        },
        "configuration": {
            "model": args.model,
            "batch_size": args.batch_size,
            "text_column": args.text_column
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / results_filename
    
    # Detailed logging about output
    LOGGER.info("=== FINAL OUTPUT INFO ===")
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info("Output directory exists: %s", output_dir.exists())
    
    # List what we've created in output
    if output_dir.exists():
        LOGGER.info("Contents of output directory:")
        for item in output_dir.iterdir():
            size = item.stat().st_size if item.is_file() else "DIR"
            LOGGER.info("  - %s (%s bytes)", item, size)
    
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    
    LOGGER.info("Results file written: %s (%d bytes)", results_file, results_file.stat().st_size)
    
    # Log performance summary
    LOGGER.info("=== PERFORMANCE SUMMARY ===")
    LOGGER.info("Total tokens processed: %s", f"{metrics['totals']['total_tokens_processed']:,}")
    LOGGER.info("Total chunks processed: %s", f"{metrics['totals']['total_chunks_processed']:,}")
    LOGGER.info("Files processed: %d new, %d skipped, %d total", 
               metrics['totals']['files_processed'], 
               metrics['totals']['files_skipped'],
               metrics['totals']['total_files_seen'])
    LOGGER.info("Tokens per minute: %s", f"{metrics['throughput']['tokens_per_minute']:,}")
    LOGGER.info("Peak memory usage: %s MB", metrics['memory_usage']['peak_memory_mb'])
    LOGGER.info("Memory per chunk: %s MB", metrics['memory_usage']['memory_per_chunk_mb'])
    LOGGER.info("=== END PERFORMANCE ===")
    LOGGER.info("=== END OUTPUT INFO ===")

    LOGGER.info("Done. Processed %d new files (%d skipped from checkpoint, %d total). Results written to %s", 
               processed, metrics['totals']['files_skipped'], processed + metrics['totals']['files_skipped'], results_file)
    
    # Force final cleanup, but in a SageMaker-friendly way
    cleanup_memory()
    
    # Log checkpoint status in final message
    checkpoint_files = len(checkpoint_mgr.get_processed_files())
    LOGGER.info("Checkpoint status: %d files in checkpoint. Checkpoint location: %s", 
               checkpoint_files, checkpoint_mgr.checkpoint_file)
    LOGGER.info("Script completed successfully - normal termination")
    
    # Create a success marker file
    try:
        success_marker = output_dir / "_SUCCESS"
        with open(success_marker, 'w') as marker:
            marker.write(f"Processing completed at {datetime.now().isoformat()}")
        LOGGER.info("Created success marker file: %s", success_marker)
    except Exception as e:
        LOGGER.info("Could not write success marker: %s", e)
    
    # Important: Use regular sys.exit() instead of os._exit()
    # This allows the container to clean up properly
    LOGGER.info("=== PROCESSING COMPLETE - EXITING NORMALLY ===")
    sys.exit(0)  # Exit with success code


if __name__ == "__main__":
    main()