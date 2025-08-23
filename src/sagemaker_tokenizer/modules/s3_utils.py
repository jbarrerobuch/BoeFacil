"""
S3 related utilities for direct file uploads and environment detection
"""
import os
import re
import json
import logging
import boto3
from pathlib import Path

LOGGER = logging.getLogger("tokenizer")

def upload_to_s3(local_file: Path, bucket: str, key: str, logger=None) -> bool:
    """Upload a file to S3 bucket

    Args:
        local_file (Path): Local file path
        bucket (str): S3 bucket name
        key (str): S3 key (path in bucket)
        logger (Logger, optional): Logger to use. Defaults to None.

    Returns:
        bool: True if successful, False otherwise
    """
    if logger is None:
        logger = LOGGER
    
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_file), bucket, key)
        logger.info(f"✓ Uploaded {local_file} to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to upload to S3: {e}")
        return False


def extract_job_name_from_env() -> str:
    """Extract the job name from SageMaker environment variables or path

    Returns:
        str: Job name or 'unknown-job' if not found
    """
    # Try to get job name from environment variable
    job_name = os.environ.get('TRAINING_JOB_NAME', '')
    
    if job_name:
        return job_name
    
    # Try to get job name from resourceconfig.json file
    try:
        with open('/opt/ml/input/config/resourceconfig.json', 'r') as f:
            config = json.load(f)
            job_name = config.get('TrainingJobName', '')
            if job_name:
                return job_name
    except Exception:
        pass
    
    # Try to extract from path - SageMaker often includes job name in paths
    try:
        # Look at checkpoint dir path, which often contains job name
        checkpoint_path = os.environ.get('SM_CHANNEL_CHECKPOINT', '')
        if checkpoint_path:
            # Extract job name from path like /opt/ml/checkpoints/job-name-1234
            match = re.search(r'/([^/]+)/?$', checkpoint_path)
            if match:
                return match.group(1)
    except Exception:
        pass
        
    # Default name with timestamp if everything fails
    from datetime import datetime
    return f"unknown-job-{datetime.now().strftime('%Y%m%d%H%M%S')}"


def get_sagemaker_s3_output_info(logger=None):
    """Extract SageMaker S3 output information from environment variables.
    
    Returns:
        tuple: (s3_bucket, s3_prefix) if available, or (None, None) if not
    """
    if logger is None:
        logger = LOGGER
        
    # Try to get output location from SageMaker environment variables
    s3_output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '')
    
    # If SM_OUTPUT_DATA_DIR doesn't exist or isn't an S3 path, try to build from other env vars
    if not s3_output_path.startswith('s3://'):
        # Try to construct from SM_OUTPUT_S3_URI if available
        s3_output_path = os.environ.get('SM_OUTPUT_S3_URI', '')
    
    if s3_output_path.startswith('s3://'):
        # Extract bucket and key from s3://bucket/key format
        s3_path_parts = s3_output_path.replace('s3://', '').split('/', 1)
        s3_bucket = s3_path_parts[0]
        s3_prefix = s3_path_parts[1] if len(s3_path_parts) > 1 else ""
        
        # Remove trailing slashes for consistency
        s3_prefix = s3_prefix.rstrip('/')
        
        logger.info(f"Detected S3 output path: s3://{s3_bucket}/{s3_prefix}")
        return s3_bucket, s3_prefix
    
    # Fallback to custom environment variables
    s3_bucket = os.environ.get('SM_OUTPUT_S3_BUCKET', '')
    s3_prefix = os.environ.get('SM_OUTPUT_S3_PREFIX', '')
    
    if s3_bucket:
        logger.info(f"Using custom S3 output: s3://{s3_bucket}/{s3_prefix}")
        return s3_bucket, s3_prefix
    
    # No S3 output information available
    logger.warning("Could not determine S3 output location from environment variables")
    return None, None
