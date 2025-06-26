#!/usr/bin/env python3
"""
Test script to verify ONNX and PyTorch backends work correctly
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add project paths
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.convnext_wrapper import ConvNextWrapper

def test_backends():
    """Test both PyTorch and ONNX backends"""
    
    model_path = "models/model_opt_S.pth"
    onnx_path = "models/model_opt_S_optimized.onnx"
    
    print("🧪 Testing ConvNeXt Backends...")
    
    # Create wrapper
    wrapper = ConvNextWrapper(model_path, input_size=256, output_size=32)
    
    # Test image (dummy)
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = [100, 100, 200, 300]  # x, y, w, h
    
    print("\n1️⃣ Testing PyTorch Backend...")
    try:
        success = wrapper.load_pytorch_model(joint_num=18)
        if success:
            print("✅ PyTorch model loaded successfully")
            pose_2d, pose_3d = wrapper.predict_pose_full(test_img, test_bbox, backend='pytorch')
            if pose_2d is not None and pose_3d is not None:
                print(f"✅ PyTorch inference successful: pose_2d shape={pose_2d.shape}, pose_3d shape={pose_3d.shape}")
            else:
                print("❌ PyTorch inference failed")
        else:
            print("❌ PyTorch model loading failed")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
    
    print("\n2️⃣ Testing ONNX Backend...")
    try:
        success = wrapper.load_onnx_model(onnx_path)
        if success:
            print("✅ ONNX model loaded successfully")
            pose_2d, pose_3d = wrapper.predict_pose_full(test_img, test_bbox, backend='onnx')
            if pose_2d is not None and pose_3d is not None:
                print(f"✅ ONNX inference successful: pose_2d shape={pose_2d.shape}, pose_3d shape={pose_3d.shape}")
            else:
                print("❌ ONNX inference failed")
        else:
            print("❌ ONNX model loading failed")
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
    
    print("\n3️⃣ Testing Fallback Pose...")
    try:
        pose_2d, pose_3d = wrapper._fallback_pose(test_img, test_bbox)
        if pose_2d is not None and pose_3d is not None:
            print(f"✅ Fallback pose successful: pose_2d shape={pose_2d.shape}, pose_3d shape={pose_3d.shape}")
        else:
            print("❌ Fallback pose failed")
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
    
    print("\n🎉 Backend testing complete!")

if __name__ == "__main__":
    test_backends()
