#!/usr/bin/env python3
"""
Test script to verify OpenPose model download and loading fix
"""

import os
import sys
import numpy as np
from PIL import Image

def test_openpose():
    """Test OpenPose model download and loading"""
    print("🧪 Testing OpenPose model fix...")
    
    try:
        # Import the fixed OpenPose module
        from ldm.modules.extra_condition.openpose.api import OpenposeInference
        
        print("✅ OpenPose import successful")
        
        # Try to initialize the model (this will trigger download if needed)
        print("📥 Initializing OpenPose model (may download if needed)...")
        pose_model = OpenposeInference()
        
        print("✅ OpenPose model initialization successful")
        
        # Create a dummy test image
        print("🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (512, 384, 3), dtype=np.uint8)
        
        # Test inference
        print("🚀 Running OpenPose inference...")
        result = pose_model(test_image)
        
        print(f"✅ OpenPose inference successful! Output shape: {result.shape}")
        print("🎉 All OpenPose tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenPose test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("OpenPose Fix Verification Test")
    print("=" * 60)
    
    success = test_openpose()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: OpenPose is working correctly!")
        print("The API should now be able to use OpenPose for pose detection.")
    else:
        print("❌ FAILED: OpenPose is still not working.")
        print("Check the error messages above for troubleshooting.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 