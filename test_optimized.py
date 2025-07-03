#!/usr/bin/env python3
"""
Quick test script to verify TryOn-Adapter optimizations work
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
from PIL import Image
import numpy as np

def create_test_images():
    """Create simple test images for TryOn-Adapter"""
    # Create a simple person image (256x192)
    person_img = Image.new('RGB', (192, 256), color='lightblue')
    person_img.save('test_person.jpg')
    
    # Create a simple clothing image (256x192)
    cloth_img = Image.new('RGB', (192, 256), color='red')
    cloth_img.save('test_cloth.jpg')
    
    # Create a simple mask
    mask_img = Image.new('L', (192, 256), color=128)
    mask_img.save('test_mask.png')
    
    return 'test_person.jpg', 'test_cloth.jpg', 'test_mask.png'

def test_optimized_inference():
    """Test the optimized TryOn-Adapter inference"""
    print("🧪 Testing optimized TryOn-Adapter inference...")
    
    # Create test images
    person_path, cloth_path, mask_path = create_test_images()
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    output_path = 'test_output/result.png'
    
    # Test command
    cmd = [
        sys.executable, 'test_viton.py',
        '--person_image', person_path,
        '--cloth_image', cloth_path,
        '--mask', mask_path,
        '--output', output_path,
        '--config', 'configs/viton.yaml',
        '--ckpt', 'checkpoints/model.ckpt',
        '--ckpt_elbm_path', 'models/vae/dirs/viton_hd_blend',
        '--H', '256',
        '--W', '192'
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for test
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  Inference completed in {elapsed_time:.2f} seconds")
        print(f"📊 Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Test PASSED - Optimized inference completed successfully!")
            if os.path.exists(output_path):
                print(f"📸 Result saved to: {output_path}")
            else:
                print("⚠️  Warning: Output file not found")
        else:
            print("❌ Test FAILED")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Test TIMEOUT - Inference took too long")
    except Exception as e:
        print(f"❌ Test ERROR: {e}")
    
    # Cleanup
    try:
        os.remove(person_path)
        os.remove(cloth_path)
        os.remove(mask_path)
        shutil.rmtree('test_output', ignore_errors=True)
    except:
        pass

if __name__ == "__main__":
    test_optimized_inference() 