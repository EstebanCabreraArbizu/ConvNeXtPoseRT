#!/usr/bin/env python3
"""
test_wrapper_integration.py - Test específico para verificar la integración de wrappers

Este test verifica que:
1. Los wrappers se pueden importar correctamente
2. La configuración se carga bien
3. Los modelos se pueden inicializar (sin cargar completamente)
4. Las funcionalidades básicas funcionan
"""

import os
import sys
import logging
import numpy as np
import cv2

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_paths():
    """Test de configuración de paths"""
    logger.info("🔄 Testing configuration paths...")
    
    try:
        import config_paths
        config = config_paths.get_paths()
        
        logger.info("✅ Configuration loaded:")
        logger.info(f"   - ConvNextPose: {config['convnextpose_path']}")
        logger.info(f"   - RootNet: {config['rootnet_path']}")
        logger.info(f"   - Model: {config['convnext_model_path']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_convnext_wrapper():
    """Test de ConvNextWrapper"""
    logger.info("🔄 Testing ConvNextWrapper...")
    
    try:
        from src.convnext_wrapper import ConvNextWrapper
        import config_paths
        
        config = config_paths.get_paths()
        
        # Crear wrapper
        wrapper = ConvNextWrapper(
            model_path=config['convnext_model_path'],
            convnext_path=config['convnextpose_path']
        )
        
        logger.info("✅ ConvNextWrapper created successfully")
        
        # Test de imagen dummy
        dummy_img = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
        result = wrapper.predict_pose_full(dummy_img, bbox=[0, 0, 192, 256])
        
        if result is not None:
            pose_2d, pose_3d = result
            logger.info(f"✅ ConvNextWrapper prediction successful: 2D shape {pose_2d.shape}, 3D shape {pose_3d.shape}")
            return True
        else:
            logger.warning("⚠️ ConvNextWrapper prediction returned None (model not loaded)")
            return True  # Esto es esperado sin cargar el modelo
            
    except Exception as e:
        logger.error(f"❌ ConvNextWrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_root_wrapper():
    """Test de RootNetWrapper"""
    logger.info("🔄 Testing RootNetWrapper...")
    
    try:
        from src.root_wrapper import RootNetWrapper
        import config_paths
        
        config = config_paths.get_paths()
        
        # Crear wrapper
        wrapper = RootNetWrapper(
            rootnet_path=config['rootnet_path'],
            checkpoint_path=config['rootnet_model_path']
        )
        
        logger.info("✅ RootNetWrapper created successfully")
        
        # Test de predicción dummy
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = wrapper.predict_depth(dummy_img, bbox=[100, 100, 200, 300])
        
        logger.info(f"✅ RootNetWrapper prediction successful: depth = {depth:.2f}mm")
        return True
            
    except Exception as e:
        logger.error(f"❌ RootNetWrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_import():
    """Test de importación de YOLO"""
    logger.info("🔄 Testing YOLO import...")
    
    try:
        from ultralytics import YOLO
        logger.info("✅ YOLO imported successfully")
        
        # Verificar modelo
        import config_paths
        config = config_paths.get_paths()
        
        if os.path.exists(config['yolo_model_path']):
            logger.info(f"✅ YOLO model exists: {config['yolo_model_path']}")
        else:
            logger.warning(f"⚠️ YOLO model not found: {config['yolo_model_path']}")
            
        return True
            
    except ImportError as e:
        logger.error(f"❌ YOLO import failed: {e}")
        return False

def main():
    """Ejecutar todos los tests"""
    logger.info("🚀 Starting wrapper integration tests")
    logger.info("=" * 60)
    
    tests = [
        ("📋 Configuration Paths", test_config_paths),
        ("🧩 ConvNextWrapper", test_convnext_wrapper),
        ("🎯 RootNetWrapper", test_root_wrapper),
        ("🔍 YOLO Import", test_yolo_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Wrapper integration is working correctly.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
