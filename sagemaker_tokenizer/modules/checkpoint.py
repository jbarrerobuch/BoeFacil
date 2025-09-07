"""
Checkpoint management module for spot instance resilience
"""
import atexit
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Set

LOGGER = logging.getLogger("tokenizer")

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
