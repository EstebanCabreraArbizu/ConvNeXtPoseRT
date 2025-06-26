#!/usr/bin/env python3
"""
test_3d_integration.py - Test script for 3D pose integration with RootNet

Tests both 2D and 3D modes to verify the integration works correctly.
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required imports work"""
    logger.info("🔍 Testing imports...")
    
    try:
        import cv2
        logger.info("✅ OpenCV available")
    except ImportError:
        logger.error("❌ OpenCV not available")
        return False
    
    try:
        import torch
        logger.info(f"✅ PyTorch available: {torch.__version__}")
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False
    
    try:
        from ultralytics import YOLO
        logger.info("✅ YOLO available")
    except ImportError:
        logger.warning("⚠️ YOLO not available")
    
    try:
        from src.root_wrapper import RootNetWrapper
        logger.info("✅ RootNet wrapper available")
    except ImportError:
        logger.warning("⚠️ RootNet wrapper not available")
    
    return True

def test_processor_initialization():
    """Test processor initialization in both 2D and 3D modes"""
    logger.info("🔍 Testing processor initialization...")
    
    try:
        from main import ProductionV4Processor
        
        # Test 2D mode
        logger.info("Testing 2D mode...")
        processor_2d = ProductionV4Processor(
            model_path='models/model_opt_S.pth',
            preset='ultra_fast',
            backend='pytorch',
            enable_3d=False
        )
        logger.info("✅ 2D processor initialized successfully")
        
        # Test 3D mode
        logger.info("Testing 3D mode...")
        processor_3d = ProductionV4Processor(
            model_path='models/model_opt_S.pth',
            preset='ultra_fast',
            backend='pytorch',
            enable_3d=True
        )
        logger.info(f"✅ 3D processor initialized: 3D enabled = {processor_3d.enable_3d}")
        
        # Cleanup
        processor_2d.cleanup()
        processor_3d.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dummy_inference():
    """Test inference with a dummy frame"""
    logger.info("🔍 Testing dummy inference...")
    
    try:
        import numpy as np
        import cv2
        from main import ProductionV4Processor
        
        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test 2D mode
        processor_2d = ProductionV4Processor(
            model_path='models/model_opt_S.pth',
            preset='ultra_fast',
            backend='pytorch',
            enable_3d=False
        )
        
        start_time = time.time()
        poses, stats = processor_2d.process_frame(dummy_frame)
        processing_time = time.time() - start_time
        
        logger.info(f"✅ 2D inference completed:")
        logger.info(f"   Processing time: {processing_time*1000:.1f}ms")
        logger.info(f"   Poses detected: {len(poses)}")
        logger.info(f"   FPS: {stats['avg_fps']:.1f}")
        
        # Test 3D mode if available
        processor_3d = ProductionV4Processor(
            model_path='models/model_opt_S.pth',
            preset='ultra_fast',
            backend='pytorch',
            enable_3d=True
        )
        
        if processor_3d.enable_3d:
            start_time = time.time()
            poses, stats = processor_3d.process_frame(dummy_frame)
            processing_time = time.time() - start_time
            
            logger.info(f"✅ 3D inference completed:")
            logger.info(f"   Processing time: {processing_time*1000:.1f}ms")
            logger.info(f"   Poses detected: {len(poses)}")
            logger.info(f"   FPS: {stats['avg_fps']:.1f}")
            logger.info(f"   RootNet active: {stats.get('rootnet_active', False)}")
        else:
            logger.warning("⚠️ 3D mode not available, testing 2D only")
        
        # Cleanup
        processor_2d.cleanup()
        processor_3d.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dummy inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_files():
    """Test if required model files are available"""
    logger.info("🔍 Testing model files...")
    
    # Check ConvNeXt model
    convnext_model = Path('models/model_opt_S.pth')
    if convnext_model.exists():
        logger.info(f"✅ ConvNeXt model found: {convnext_model}")
    else:
        logger.warning(f"⚠️ ConvNeXt model not found: {convnext_model}")
    
    # Check YOLO model
    yolo_model = Path('models/yolo11n.pt')
    if yolo_model.exists():
        logger.info(f"✅ YOLO model found: {yolo_model}")
    else:
        logger.warning(f"⚠️ YOLO model not found: {yolo_model}")
    
    # Check ONNX models
    onnx_models = [
        Path('models/model_opt_S_optimized.onnx'),
        Path('models/model_opt_S.onnx'),
        Path('models/model_S.onnx')
    ]
    
    onnx_found = False
    for onnx_model in onnx_models:
        if onnx_model.exists():
            logger.info(f"✅ ONNX model found: {onnx_model}")
            onnx_found = True
            break
    
    if not onnx_found:
        logger.warning("⚠️ No ONNX models found")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test 3D pose integration')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference tests (faster)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting 3D pose integration tests...")
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files),
        ("Processor Initialization", test_processor_initialization),
    ]
    
    if not args.skip_inference:
        tests.append(("Dummy Inference", test_dummy_inference))
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        success = test_func()
        results.append((test_name, success))
        
        if success:
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! 3D integration ready.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
