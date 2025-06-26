# Configuración para ConvNeXtPoseRT

# Path absoluto a ConvNextPose - DEBES CONFIGURAR ESTO PARA TU SISTEMA
# Ejemplo: CONVNEXT_POSE_PATH = "C:/Users/usuario/ConvNextPose"
# Ejemplo: CONVNEXT_POSE_PATH = "/home/usuario/ConvNextPose"
CONVNEXT_POSE_PATH = "D:/Repository-Projects/ConvNeXtPose"  # ⚠️ CAMBIAR ESTO

# Path absoluto a RootNet - DEBES CONFIGURAR ESTO PARA TU SISTEMA
# Ejemplo: ROOTNET_PATH = "C:/Users/usuario/RootNet"
# Ejemplo: ROOTNET_PATH = "/home/usuario/RootNet"
ROOTNET_PATH = "D:/Repository-Projects/3DMPPE_ROOTNET_RELEASE"  # ⚠️ CAMBIAR ESTO
ROOTNET_MODEL_PATH = "models/snapshot_19.pth.tar"  # Relativo al directorio del proyecto
# Configuración del modelo
MODEL_PATH = "models/model_opt_S.pth"  # Relativo al directorio del proyecto

# Configuración de detección
YOLO_MODEL_PATH = "models/yolo11n.pt"  # Relativo al directorio del proyecto

# Configuración de backends
ENABLE_ONNX = True
ENABLE_TFLITE = True

# Configuración de logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR

# Configuración de hardware
FORCE_CPU = False  # Forzar uso de CPU incluso si CUDA está disponible
NUM_THREADS = 4    # Número de threads para CPU

def get_paths():
    """Obtiene los paths de configuración como diccionario"""
    import os
    
    # Obtener directorio base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    return {
        'convnextpose_path': CONVNEXT_POSE_PATH,
        'rootnet_path': ROOTNET_PATH,
        'rootnet_model_path': os.path.join(base_dir, ROOTNET_MODEL_PATH),
        'convnext_model_path': os.path.join(base_dir, MODEL_PATH),
        'yolo_model_path': os.path.join(base_dir, YOLO_MODEL_PATH),
        'enable_onnx': ENABLE_ONNX,
        'enable_tflite': ENABLE_TFLITE,
        'log_level': LOG_LEVEL,
        'force_cpu': FORCE_CPU,
        'num_threads': NUM_THREADS,
        'base_dir': base_dir
    }
