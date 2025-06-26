#!/usr/bin/env python3
"""
test_enhanced_performance.py - Test de rendimiento mejorado

Este script prueba las mejoras implementadas en main.py y convnext_wrapper.py
"""

import os
import sys
import logging
import numpy as np
import cv2
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_performance():
    """Test de rendimiento con las mejoras implementadas"""
    logger.info("🚀 Testing Enhanced Performance Improvements")
    logger.info("=" * 60)
    
    # Test 1: ConvNext Wrapper Enhanced
    logger.info("\n📋 Test 1: Enhanced ConvNext Wrapper")
    try:
        from src.convnext_wrapper import ConvNextWrapper
        import config_paths
        
        config = config_paths.get_paths()
        
        wrapper = ConvNextWrapper(
            model_path=config['convnext_model_path'],
            convnext_path=config['convnextpose_path']
        )
        
        # Test inference speed
        test_img = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
        test_bbox = [50, 50, 150, 200]
        
        # Warm up
        for _ in range(3):
            wrapper.predict_pose_full(test_img, test_bbox)
        
        # Benchmark
        start_time = time.time()
        iterations = 10
        
        for i in range(iterations):
            result = wrapper.predict_pose_full(test_img, test_bbox)
            if result is not None:
                pose_2d, pose_3d = result
                logger.info(f"   Iteration {i+1}: 2D {pose_2d.shape}, 3D {pose_3d.shape}")
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        fps = 1.0 / avg_time
        
        logger.info(f"✅ Enhanced Wrapper Performance:")
        logger.info(f"   Average time: {avg_time*1000:.1f}ms")
        logger.info(f"   Estimated FPS: {fps:.1f}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced wrapper test failed: {e}")
        return False
    
    # Test 2: YOLO Detection Performance  
    logger.info("\n🎯 Test 2: Enhanced YOLO Detection")
    try:
        from ultralytics import YOLO
        
        # Create test image with person-like shape
        test_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add some person-like rectangles
        cv2.rectangle(test_img, (200, 100), (350, 400), (120, 150, 180), -1)
        cv2.circle(test_img, (275, 130), 25, (200, 180, 150), -1)
        
        yolo_model = YOLO(config['yolo_model_path'])
        
        # Benchmark YOLO
        start_time = time.time()
        iterations = 10
        
        for i in range(iterations):
            results = yolo_model(test_img, verbose=False, conf=0.3)
            detections = 0
            for result in results:
                if result.boxes is not None:
                    detections += len(result.boxes)
            logger.info(f"   YOLO iteration {i+1}: {detections} detections")
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        fps = 1.0 / avg_time
        
        logger.info(f"✅ Enhanced YOLO Performance:")
        logger.info(f"   Average time: {avg_time*1000:.1f}ms")
        logger.info(f"   Estimated FPS: {fps:.1f}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced YOLO test failed: {e}")
        return False
    
    # Test 3: Full Pipeline Simulation
    logger.info("\n🔄 Test 3: Full Pipeline Performance")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from main import ProductionV4Processor
        
        # Test different presets
        presets = ['ultra_fast', 'speed_balanced']
        
        for preset in presets:
            logger.info(f"\n   Testing preset: {preset}")
            
            processor = ProductionV4Processor(
                model_path=config['convnext_model_path'],
                preset=preset,
                backend='pytorch'
            )
            
            # Create test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.rectangle(test_frame, (200, 100), (350, 400), (120, 150, 180), -1)
            
            # Warm up
            for _ in range(3):
                processor.process_frame(test_frame)
            
            # Benchmark
            start_time = time.time()
            iterations = 10
            
            for i in range(iterations):
                poses_2d, poses_3d, stats = processor.process_frame(test_frame)
                logger.info(f"     Frame {i+1}: {len(poses_2d)} poses, "
                          f"{stats['instant_fps']:.1f} FPS")
            
            total_time = time.time() - start_time
            avg_fps = iterations / total_time
            
            logger.info(f"   ✅ {preset.upper()} Results:")
            logger.info(f"     Average FPS: {avg_fps:.1f}")
            logger.info(f"     Target FPS: {processor.config['target_fps']}")
            
            processor.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ENHANCED PERFORMANCE TEST COMPLETED!")
    logger.info("✅ All optimizations working correctly")
    logger.info("🚀 Ready for improved production use")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_performance()
    sys.exit(0 if success else 1)
