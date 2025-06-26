#!/usr/bin/env python3
"""
test_wrapper.py - Script de prueba para ConvNextWrapper

Este script verifica que el wrapper funciona correctamente sin ejecutar
el pipeline completo de detección y pose.
"""

import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from convnext_wrapper import ConvNextWrapper
    logger.info("✅ ConvNextWrapper importado correctamente")
except ImportError as e:
    logger.error(f"❌ Error al importar ConvNextWrapper: {e}")
    sys.exit(1)

def test_wrapper_initialization():
    """Probar la inicialización del wrapper"""
    logger.info("🔄 Probando inicialización del wrapper...")
    
    # Configuración de prueba
    model_path = "models/model_opt_S.pth"
    convnext_path = None  # Auto-detectar o configurar manualmente
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        logger.info("💡 Asegúrate de que el modelo esté en la carpeta 'models/'")
        return False
    
    try:
        # Inicializar wrapper
        wrapper = ConvNextWrapper(
            model_path=model_path,
            input_size=256,
            output_size=32,
            convnext_path=convnext_path  # Auto-detectar
        )
        
        logger.info("✅ Wrapper inicializado correctamente")
        logger.info(f"   - Modelo: {model_path}")
        logger.info(f"   - ConvNextPose path: {wrapper.convnext_path}")
        logger.info(f"   - Device: {wrapper.device}")
        
        return wrapper
        
    except Exception as e:
        logger.error(f"❌ Error al inicializar wrapper: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_pytorch_loading(wrapper):
    """Probar carga del modelo PyTorch"""
    logger.info("🔄 Probando carga del modelo PyTorch...")
    
    if wrapper.convnext_path is None:
        logger.error("❌ ConvNextPose path no detectado")
        logger.info("💡 Configura manualmente el path en config_paths.py")
        return False
    
    try:
        # Intentar cargar modelo PyTorch
        success = wrapper.load_pytorch_model(joint_num=18)
        
        if success:
            logger.info("✅ Modelo PyTorch cargado correctamente")
            logger.info(f"   - Transform: {wrapper.transform is not None}")
            logger.info(f"   - Config: {wrapper.cfg is not None}")
            return True
        else:
            logger.error("❌ Falló la carga del modelo PyTorch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo PyTorch: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dummy_inference(wrapper):
    """Probar inferencia con datos dummy"""
    logger.info("🔄 Probando inferencia con datos dummy...")
    
    try:
        # Crear imagen dummy (256x256x3)
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Probar inferencia PyTorch
        output = wrapper.infer_pytorch(dummy_img)
        
        if output is not None:
            logger.info("✅ Inferencia PyTorch exitosa")
            logger.info(f"   - Output shape: {output.shape}")
            logger.info(f"   - Output dtype: {output.dtype}")
            return True
        else:
            logger.error("❌ Inferencia PyTorch devolvió None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en inferencia: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_onnx_loading(wrapper):
    """Probar carga de modelo ONNX (opcional)"""
    logger.info("🔄 Probando carga de modelo ONNX...")
    
    # Buscar modelo ONNX
    onnx_candidates = [
        "models/model_opt_S_optimized.onnx",
        "models/model_S.onnx", 
        "models/model_opt_S.onnx"
    ]
    
    onnx_path = None
    for candidate in onnx_candidates:
        if os.path.exists(candidate):
            onnx_path = candidate
            break
    
    if onnx_path is None:
        logger.warning("⚠️ No se encontró modelo ONNX - saltando prueba")
        return True  # No es un error crítico
    
    try:
        success = wrapper.load_onnx_model(onnx_path)
        
        if success:
            logger.info("✅ Modelo ONNX cargado correctamente")
            logger.info(f"   - Path: {onnx_path}")
            return True
        else:
            logger.warning("⚠️ Falló la carga del modelo ONNX")
            return True  # No es crítico
            
    except Exception as e:
        logger.warning(f"⚠️ Error al cargar modelo ONNX: {e}")
        return True  # No es crítico

def main():
    """Función principal de prueba"""
    logger.info("🚀 Iniciando pruebas del ConvNextWrapper")
    logger.info("=" * 50)
    
    # Paso 1: Inicialización
    wrapper = test_wrapper_initialization()
    if wrapper is None:
        logger.error("❌ Falló la inicialización - abortando pruebas")
        return False
    
    # Paso 2: Carga PyTorch
    if not test_pytorch_loading(wrapper):
        logger.error("❌ Falló la carga PyTorch - abortando pruebas")
        return False
    
    # Paso 3: Inferencia dummy
    if not test_dummy_inference(wrapper):
        logger.error("❌ Falló la inferencia - abortando pruebas")
        return False
    
    # Paso 4: ONNX (opcional)
    test_onnx_loading(wrapper)
    
    logger.info("=" * 50)
    logger.info("✅ Todas las pruebas completadas exitosamente")
    logger.info("💡 El wrapper está listo para usar en main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
