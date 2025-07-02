#!/usr/bin/env python3
"""
Single image human parsing using CIHP_PGN
This script runs the CIHP_PGN human parser on a single input image
and saves the parsing result to the specified output path.
"""

import argparse
import os
import subprocess
import shutil
import logging
import sys

def run_human_parser(input_image_path, output_mask_path):
    """
    Run CIHP_PGN human parser on a single image
    
    Args:
        input_image_path: Path to input person image
        output_mask_path: Path where the parsing mask should be saved
    """
    # Set up paths
    HOME = os.path.expanduser("~")
    CIHP_PGN_DIR = os.path.join(HOME, "CIHP_PGN")
    CHECKPOINT_DIR = os.path.join(HOME, "checkpoint")
    DATASETS_DIR = os.path.join(HOME, "datasets")
    OUTPUT_DIR = os.path.join(HOME, "output")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Input image path: {input_image_path}")
    logger.info(f"Output mask path: {output_mask_path}")
    logger.info(f"CIHP_PGN_DIR: {CIHP_PGN_DIR}")

    # Check if input image exists
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    # Check if CIHP_PGN directory exists
    if not os.path.exists(CIHP_PGN_DIR):
        logger.warning(f"CIHP_PGN directory not found at {CIHP_PGN_DIR}")
        logger.warning("Human parsing will not be available")
        return False

    try:
        # Prepare directories
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Copy input image to expected location
        input_copy_path = os.path.join(DATASETS_DIR, "input.jpg")
        shutil.copy2(input_image_path, input_copy_path)
        logger.info(f"Copied input image to {input_copy_path}")

        # Run CIHP_PGN human parser in conda environment
        command = ["conda", "run", "-n", "cihp_pgn", "python", "test_pgn.py"]
        logger.info(f"Running CIHP_PGN human parser: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            cwd=CIHP_PGN_DIR,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        logger.info(f"Human parser return code: {result.returncode}")
        if result.stdout:
            logger.info(f"STDOUT: {result.stdout}")
        if result.stderr:
            logger.info(f"STDERR: {result.stderr}")
            
        if result.returncode != 0:
            logger.error("CIHP_PGN parser failed!")
            return False

        # Check if output was created
        mask_path = os.path.join(OUTPUT_DIR, "input.png")
        logger.info(f"Looking for mask at {mask_path}")
        
        if not os.path.exists(mask_path):
            logger.error(f"Parser output not found at {mask_path}")
            # List files in output directory for debugging
            if os.path.exists(OUTPUT_DIR):
                files = os.listdir(OUTPUT_DIR)
                logger.error(f"Files in output dir: {files}")
            return False

        # Copy result to final output path
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        shutil.copy2(mask_path, output_mask_path)
        logger.info(f"Human parsing mask saved to: {output_mask_path}")

        # Clean up temporary files
        try:
            os.remove(input_copy_path)
            os.remove(mask_path)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Human parser timed out after 2 minutes")
        return False
    except Exception as e:
        logger.error(f"Human parser failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run CIHP_PGN human parser on a single image')
    parser.add_argument('--input', required=True, help='Path to input person image')
    parser.add_argument('--output', required=True, help='Path to save parsing mask')
    
    args = parser.parse_args()
    
    print(f"Running human parser on: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    success = run_human_parser(args.input, args.output)
    
    if success:
        print("Human parsing completed successfully!")
        sys.exit(0)
    else:
        print("Human parsing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 