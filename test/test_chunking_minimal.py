#!/usr/bin/env python3
"""
Minimal test script for chunking functionality without loading all parquet files
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Set up the path to import from lib
import sys
sys.path.append(str(Path(__file__).parent / "src"))

# Import the chunking function
from lib.chunk_utils import chunking_markdown_df

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_dataframe():
    """Create a small test DataFrame with markdown content"""
    test_data = {
        'item_id': ['test_001', 'test_002', 'test_003'],
        'markdown': [
            '# Test Document 1\n\nThis is a test markdown document with some content. ' * 50,  # Long content
            '# Test Document 2\n\nAnother test document with different content. ' * 30,  # Medium content  
            '# Short Document\n\nBrief content.'  # Short content
        ]
    }
    return pd.DataFrame(test_data)

def main():
    """Test the chunking functionality with a small dataset"""
    
    # Define paths
    base_path = Path(__file__).parent
    chunks_dir = base_path / "samples" / "chunks" / "test"
    
    logger.info(f"Testing chunking functionality...")
    logger.info(f"Chunks will be saved to: {chunks_dir}")
    
    # Create test DataFrame
    test_df = create_test_dataframe()
    logger.info(f"Created test DataFrame with {len(test_df)} rows")
    logger.info(f"Columns: {list(test_df.columns)}")
    
    # Test chunking
    try:
        chunks = chunking_markdown_df(
            df=test_df,
            columna_markdown='markdown',
            max_tokens=100,  # Small chunks for testing
            texto_id=None,  # Use item_id from DataFrame
            store_path=str(chunks_dir)
        )
        
        logger.info(f"✅ Chunking completed successfully!")
        logger.info(f"Generated {len(chunks)} chunks total")
        
        if chunks:
            chunks_guardados = sum(1 for chunk in chunks if 'archivo_guardado' in chunk)
            logger.info(f"Chunks saved to files: {chunks_guardados}")
            
            # Show statistics
            tokens_stats = [chunk['tokens_aproximados'] for chunk in chunks]
            logger.info(f"Token stats - Avg: {sum(tokens_stats)/len(tokens_stats):.1f}, "
                       f"Min: {min(tokens_stats)}, Max: {max(tokens_stats)}")
            
            # Show sample chunk info
            logger.info(f"Sample chunk info:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                logger.info(f"  Chunk {i}: {chunk['tokens_aproximados']} tokens, "
                           f"from row {chunk['fila_original']}, "
                           f"item_id: {chunk.get('item_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during chunking: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
