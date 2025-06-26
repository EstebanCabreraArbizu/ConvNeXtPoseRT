#!/usr/bin/env python3
"""
test_main_real.py - Test del main.py original con imagen simulada

Este test verifica que main.py funciona con una imagen generada
"""

import os
import sys
import cv2
import numpy as np
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_video_frame():
    """Crear frame de video de prueba"""
    width, height = 640, 480
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Simular figura humana más realista
    person_x = width // 2 - 60
    person_y = height // 2 - 120
    person_w = 120
    person_h = 240
    
    # Cuerpo
    cv2.rectangle(img, (person_x, person_y + 40), (person_x + person_w, person_y + person_h), (120, 150, 180), -1)
    
    # Cabeza
    cv2.circle(img, (person_x + person_w//2, person_y + 25), 20, (200, 180, 150), -1)
    
    # Brazos
    cv2.rectangle(img, (person_x - 15, person_y + 60), (person_x + 15, person_y + 140), (100, 130, 160), -1)
    cv2.rectangle(img, (person_x + person_w - 15, person_y + 60), (person_x + person_w + 15, person_y + 140), (100, 130, 160), -1)
    
    # Piernas
    cv2.rectangle(img, (person_x + 20, person_y + 180), (person_x + 50, person_y + person_h + 30), (90, 120, 150), -1)
    cv2.rectangle(img, (person_x + 70, person_y + 180), (person_x + 100, person_y + person_h + 30), (90, 120, 150), -1)
    
    logger.info(f"✅ Created realistic test frame: {width}x{height}")
    return img

def test_main_with_simulated_input():
    """Test main.py con entrada simulada"""
    logger.info("🚀 Testing main.py with simulated input")
    
    # Crear imagen de prueba
    test_img = create_test_video_frame()
    
    # Guardar imagen temporal
    temp_img_path = "temp_test_frame.jpg"
    cv2.imwrite(temp_img_path, test_img)
    logger.info(f"✅ Saved test image: {temp_img_path}")
    
    try:
        # Simular argumentos de línea de comandos para main.py
        test_args = [
            '--input', temp_img_path,
            '--preset', 'balanced',
            '--backend', 'pytorch',
            '--max_frames', '1',
            '--output_dir', 'test_output',
            '--save_visualization'
        ]
        
        # Configurar sys.argv para main.py
        original_argv = sys.argv.copy()
        sys.argv = ['main.py'] + test_args
        
        logger.info("🔄 Running main.py with test arguments...")
        logger.info(f"   Arguments: {' '.join(test_args)}")
        
        # Importar y ejecutar main
        import main
        
        logger.info("✅ main.py executed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ main.py test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restaurar argv y limpiar
        sys.argv = original_argv
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            logger.info("🧹 Cleaned up temp files")

if __name__ == "__main__":
    success = test_main_with_simulated_input()
    sys.exit(0 if success else 1)
