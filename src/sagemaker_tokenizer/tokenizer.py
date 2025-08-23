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
import boto3
import re

# Import utility modules
try:
    from .modules.checkpoint import CheckpointManager
    from .modules.performance import PerformanceTracker
    from .modules.utils import get_optimal_batch_size, cleanup_memory
    from .modules.s3_utils import upload_to_s3, extract_job_name_from_env, get_sagemaker_s3_output_info
except ImportError:
    # Fall back to local directory if not properly installed
    try:
        from modules.checkpoint import CheckpointManager
        from modules.performance import PerformanceTracker
        from modules.utils import get_optimal_batch_size, cleanup_memory
        from modules.s3_utils import upload_to_s3, extract_job_name_from_env, get_sagemaker_s3_output_info
    except ImportError:
        # If modules aren't found, define critical functions inline
        def upload_to_s3(local_file, bucket, key, logger=None):
            if logger is None:
                logger = logging.getLogger(__name__)
            try:
                s3_client = boto3.client('s3')
                s3_client.upload_file(str(local_file), bucket, key)
                logger.info(f"✓ Uploaded {local_file} to s3://{bucket}/{key}")
                return True
            except Exception as e:
                logger.error(f"✗ Failed to upload to S3: {e}")
                return False
                
        def extract_job_name_from_env():
            """Extract job name or generate fallback name"""
            try:
                # Try resourceconfig.json first
                if os.path.exists('/opt/ml/input/config/resourceconfig.json'):
                    with open('/opt/ml/input/config/resourceconfig.json', 'r') as f:
                        config = json.load(f)
                        job_name = config.get('TrainingJobName', '')
                        if job_name:
                            return job_name
                
                # Default with timestamp
                return f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            except Exception:
                return f"job-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
        def get_sagemaker_s3_output_info(logger=None):
            """Extract S3 bucket and prefix from SageMaker environment variables"""
            if logger is None:
                logger = logging.getLogger(__name__)
                
            try:
                s3_output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '')
                
                if s3_output_path.startswith('s3://'):
                    # Extract bucket and key from s3://bucket/key format
                    s3_path_parts = s3_output_path.replace('s3://', '').split('/', 1)
                    s3_bucket = s3_path_parts[0]
                    s3_prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
                    s3_prefix = s3_prefix.rstrip('/')
                    
                    logger.info(f"Extracted S3 output info - bucket: {s3_bucket}, prefix: {s3_prefix}")
                    return s3_bucket, s3_prefix
                else:
                    logger.warning("Could not extract S3 info - SM_OUTPUT_DATA_DIR not an S3 path")
                    return None, None
            except Exception as e:
                logger.warning(f"Failed to extract S3 output info: {e}")
                return None, None

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


LOGGER = logging.getLogger("tokenizer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


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
        
        # Also upload directly to S3 to avoid compression
        try:
            # Extract the S3 output path from Training Job's output location
            # Format is typically s3://bucket/prefix/job-name/
            s3_output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '')
            
            # If we have S3 output info, use it directly
            if s3_output_path.startswith('s3://'):
                # Extract bucket and key from s3://bucket/key format
                s3_path_parts = s3_output_path.replace('s3://', '').split('/', 1)
                s3_bucket = s3_path_parts[0]
                s3_prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
                
                # Remove trailing slashes for consistency
                s3_prefix = s3_prefix.rstrip('/')
                
                # Create S3 key for this specific file
                s3_key = f"{s3_prefix}/{input_path.name}"
            else:
                # Fallback if SM_OUTPUT_DATA_DIR isn't a proper S3 path
                s3_bucket = os.environ.get('SM_OUTPUT_S3_BUCKET', 'boe-facil')
                s3_output_prefix = os.environ.get('SM_OUTPUT_S3_PREFIX', 'test_job/output')
                job_name = extract_job_name_from_env()
                s3_key = f"{s3_output_prefix}/{job_name}/{input_path.name}"
            
            # Upload file directly to S3
            upload_to_s3(out_file, s3_bucket, s3_key, LOGGER)
            LOGGER.info(f"✓ File also uploaded directly to S3: s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            LOGGER.warning(f"Could not upload directly to S3: {e}")
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
    
    # Try to get S3 output info and set in environment
    try:
        s3_bucket, s3_prefix = get_sagemaker_s3_output_info(LOGGER)
        if s3_bucket:
            os.environ['SM_OUTPUT_S3_BUCKET'] = s3_bucket
            os.environ['SM_OUTPUT_S3_PREFIX'] = s3_prefix
    except Exception:
        LOGGER.info("Could not get S3 output info - direct S3 upload may use fallback paths")

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
    
    # Upload results file directly to S3 as well
    try:
        # Extract the S3 output path from Training Job's output location
        s3_output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '')
        
        if s3_output_path.startswith('s3://'):
            # Extract bucket and key from s3://bucket/key format
            s3_path_parts = s3_output_path.replace('s3://', '').split('/', 1)
            s3_bucket = s3_path_parts[0]
            s3_prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
            
            # Remove trailing slashes for consistency
            s3_prefix = s3_prefix.rstrip('/')
            
            # Create S3 key for results file
            s3_key = f"{s3_prefix}/{results_filename}"
            
            # Upload results file directly to S3
            upload_to_s3(results_file, s3_bucket, s3_key, LOGGER)
            LOGGER.info(f"✓ Results file also uploaded directly to S3: s3://{s3_bucket}/{s3_key}")
        else:
            LOGGER.warning("Could not determine S3 output path from environment")
    except Exception as e:
        LOGGER.warning(f"Could not upload results file to S3: {e}")
    
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