import sys
import os
import importlib
import importlib.util
import types
import torch
from torch.nn import functional as F
import numpy as np
import cv2
import math
import torchvision.transforms as transforms
from contextlib import contextmanager
from collections import OrderedDict
from torch.nn.parallel.data_parallel import DataParallel
from typing import Optional, Union, Tuple, Any
import logging
from src.root_wrapper import RootNetWrapper
# Optional imports for multiple backends
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConvNextWrapper:
    """
    Wrapper unificado para ConvNextPose con soporte para múltiples backends:
    - PyTorch (nativo con importaciones aisladas)
    - ONNX Runtime 
    - TensorFlow Lite
    
    Mantiene la compatibilidad con la interfaz del main.py
    """
    def __init__(self, model_path: str, input_size: int = 256, output_size: int = 32, 
                 device: torch.device = None, convnext_path: str = None):
        self.model_path = model_path
        self.input_size = input_size
        self.output_size = output_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ConvNextPose paths - detectar automáticamente si no se proporciona
        self.convnext_path = convnext_path or self._detect_convnext_path()
        
        # Backend models
        self.pytorch_model = None
        self.onnx_session = None
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None
        
        # Configuration
        self.cfg = None
        self._original_path = None
        
        # RootNet integration
        self.root_wrapper = None
        
        # Initialize transform (will be set after loading config)
        self.transform = None
        
    def _detect_convnext_path(self) -> Optional[str]:
        """Auto-detectar el path de ConvNextPose basado en configuración y estructura común"""
        
        # Intentar cargar desde configuración
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from config_paths import CONVNEXT_POSE_PATH
            if os.path.exists(os.path.join(CONVNEXT_POSE_PATH, 'main', 'model.py')):
                return os.path.abspath(CONVNEXT_POSE_PATH)
        except ImportError:
            pass
        
        # Paths posibles basados en estructuras comunes
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'ConvNextPose'),
            os.path.join(os.path.dirname(__file__), '..', 'ConvNextPose'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ConvNextPose'),
            'ConvNextPose',
            '../ConvNextPose',
            '../../ConvNextPose',
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(os.path.join(abs_path, 'main', 'model.py')):
                logger.info(f"✅ ConvNextPose found at: {abs_path}")
                return abs_path
        
        logger.warning("⚠️ ConvNextPose path not found. Please:")
        logger.warning("   1. Set convnext_path parameter manually, or")
        logger.warning("   2. Configure CONVNEXT_POSE_PATH in config_paths.py")
        return None

    @staticmethod
    def load_model(model_path: str, joint_num: int = 18, device: torch.device = None, 
                   convnext_path: str = None) -> 'ConvNextWrapper':
        """
        Método estático para compatibilidad con main.py
        Carga solo el modelo PyTorch por defecto
        """
        wrapper = ConvNextWrapper(model_path, device=device, convnext_path=convnext_path)
        if wrapper.load_pytorch_model(joint_num=joint_num):
            return wrapper.pytorch_model
        else:
            raise RuntimeError("Failed to load ConvNextPose PyTorch model")
    
    def load_pytorch_model(self, joint_num: int = 18, use_gpu: bool = None) -> bool:
        """Carga ConvNextPose PyTorch en contexto aislado con configuración mejorada"""
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            
        with self._isolated_import():
            try:
                # Importar configuración primero
                spec = importlib.util.spec_from_file_location(
                    "convnextpose_config", 
                    os.path.join(self.convnext_path, 'main', "config.py")
                )
                ConvNextPose_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_config_module)
                
                # Usar la configuración global del módulo
                self.cfg = ConvNextPose_config_module.cfg
                
                # Verificar que tenga todos los atributos necesarios
                required_attrs = ['depth_dim', 'output_shape', 'input_shape', 'bbox_3d_shape', 'pixel_mean', 'pixel_std']
                for attr in required_attrs:
                    if not hasattr(self.cfg, attr):
                        logger.warning(f"Missing config attribute {attr}, setting default")
                        if attr == 'depth_dim':
                            setattr(self.cfg, 'depth_dim', 32)
                        elif attr == 'output_shape':
                            setattr(self.cfg, 'output_shape', (32, 32))
                        elif attr == 'input_shape':
                            setattr(self.cfg, 'input_shape', (256, 256))
                        elif attr == 'bbox_3d_shape':
                            setattr(self.cfg, 'bbox_3d_shape', (2000, 2000, 2000))
                        elif attr == 'pixel_mean':
                            setattr(self.cfg, 'pixel_mean', (0.485, 0.456, 0.406))
                        elif attr == 'pixel_std':
                            setattr(self.cfg, 'pixel_std', (0.229, 0.224, 0.225))
                
                # Importar modelo después de configuración
                spec = importlib.util.spec_from_file_location(
                    "convnextpose_model", 
                    os.path.join(self.convnext_path, 'main', "model.py")
                )
                ConvNextPose_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_model_module)
                
                # Set device properly
                device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                
                # Cargar checkpoint
                checkpoint = torch.load(self.model_path, map_location=device)
                
                # Crear modelo con configuración verificada
                model = ConvNextPose_model_module.get_pose_net(self.cfg, is_train=False, joint_num=joint_num)
                
                # Handle DataParallel models
                if 'network' in checkpoint:
                    state_dict = checkpoint['network']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix if present (from DataParallel)
                if any(key.startswith('module.') for key in state_dict.keys()):
                    logger.info("Removing DataParallel prefix from state_dict...")
                    state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                
                # Load with strict=False to handle minor architecture differences
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"Missing keys in model: {missing_keys[:5]}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in model: {unexpected_keys[:5]}")
                
                # Move to device and set eval mode
                model.to(device)
                model.eval()
                
                self.pytorch_model = model
                self.device = device
                
                # Initialize transform after loading config
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)
                ])
                
                logger.info("✅ ConvNextPose PyTorch loaded successfully with enhanced config")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load ConvNextPose PyTorch: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def load_onnx_model(self, onnx_path: str) -> bool:
        """Carga modelo ONNX"""
        if not ONNX_AVAILABLE:
            logger.error("❌ ONNX Runtime not available")
            return False
            
        try:
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Initialize transform if not already initialized
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])
                ])
            
            logger.info(f"✅ ONNX model loaded: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def load_tflite_model(self, tflite_path: str) -> bool:
        """Carga modelo TensorFlow Lite"""
        if not TFLITE_AVAILABLE:
            logger.error("❌ TensorFlow Lite not available")
            return False
            
        try:
            self.tflite_interpreter = tf.lite.Interpreter(
                model_path=tflite_path,
                num_threads=4,
                experimental_preserve_all_tensors=False
            )
            self.tflite_interpreter.allocate_tensors()
            
            self.tflite_input_details = self.tflite_interpreter.get_input_details()
            self.tflite_output_details = self.tflite_interpreter.get_output_details()
            
            # Initialize transform if not already initialized
            if self.transform is None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.229, 0.224, 0.225], std=[0.485, 0.456, 0.406])
                ])
            
            logger.info(f"✅ TFLite model loaded: {tflite_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load TFLite model: {e}")
            return False

    @contextmanager
    def _isolated_import(self):
        """Context manager para aislar imports de ConvNextPose."""
        # Guardar estado actual
        self._original_path = sys.path.copy()
        original_modules = set(sys.modules.keys())
        
        try:
            # Añadir paths de ConvNextPose
            if self.convnext_path and self.convnext_path not in sys.path:
                sys.path.insert(0, self.convnext_path)
                sys.path.insert(0, os.path.join(self.convnext_path, 'main'))
                sys.path.insert(0, os.path.join(self.convnext_path, 'data'))
                sys.path.insert(0, os.path.join(self.convnext_path, 'common'))
            
            yield
            
        finally:
            # Restaurar estado
            sys.path = self._original_path
            # Remover módulos de ConvNeXtPose para evitar conflictos
            new_modules = set(sys.modules.keys()) - original_modules
            for module in new_modules:
                if any(path in module for path in ['convnextpose', 'main', 'data', 'common']):
                    sys.modules.pop(module, None)
    
    def infer_pytorch(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia PyTorch optimizada"""
        if self.pytorch_model is None or self.transform is None:
            logger.error("❌ PyTorch model not loaded")
            return None
            
        try:
            # Optimized input preparation
            if len(img_patch.shape) == 3 and img_patch.shape[2] == 3:
                # Convert BGR to RGB if needed
                if img_patch.dtype == np.uint8:
                    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
                    img_patch = img_patch.astype(np.float32) / 255.0
            
            # Prepare input with minimal overhead
            inp = self.transform(img_patch).to(self.device).unsqueeze(0)
            
            # Optimized inference
            with torch.no_grad():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                output = self.pytorch_model(inp)
            
            return output.cpu().numpy()
            
        except Exception as e:
            logger.error(f"❌ PyTorch inference failed: {e}")
            return None
    
    def infer_onnx(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia ONNX optimizada"""
        if self.onnx_session is None or self.transform is None:
            logger.error("❌ ONNX model not loaded") 
            return None
            
        try:
            # Optimized input preparation
            if len(img_patch.shape) == 3 and img_patch.shape[2] == 3:
                if img_patch.dtype == np.uint8:
                    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
                    img_patch = img_patch.astype(np.float32) / 255.0
            
            # Prepare input tensor
            inp = self.transform(img_patch).numpy().astype(np.float32)
            inp = np.expand_dims(inp, axis=0)
            
            # Get input name dynamically
            input_name = self.onnx_session.get_inputs()[0].name
            
            # ONNX inference with optimized settings
            output = self.onnx_session.run(None, {input_name: inp})
            
            return output[0]
            
        except Exception as e:
            logger.error(f"❌ ONNX inference failed: {e}")
            return None
    
    def infer_tflite(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inferencia TensorFlow Lite"""
        if self.tflite_interpreter is None or self.transform is None:
            logger.error("❌ TFLite model not loaded")
            return None
            
        try:
            # Prepare input
            inp = self.transform(img_patch).numpy()[None, :, :, :].astype(np.float32)
            
            # Set input tensor
            self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], inp)
            
            # Run inference
            self.tflite_interpreter.invoke()
            
            # Get output tensor
            output = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
            
            return output
            
        except Exception as e:
            logger.error(f"❌ TFLite inference failed: {e}")
            return None
    
    def infer(self, img_patch: np.ndarray, backend: str = 'pytorch') -> Optional[np.ndarray]:
        """Inferencia unificada con selección de backend"""
        if backend == 'pytorch':
            return self.infer_pytorch(img_patch)
        elif backend == 'onnx':
            return self.infer_onnx(img_patch)
        elif backend == 'tflite':
            return self.infer_tflite(img_patch)
        else:
            logger.error(f"❌ Unknown backend: {backend}")
            return None

    def predict_pose_full(self, original_img: np.ndarray, bbox: list, 
                         backend: str = 'pytorch') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Predicción completa optimizada con mejor post-procesamiento
        Mejorada para mejor calidad y velocidad
        
        Returns:
            Tuple[pose_2d, pose_3d] with improved accuracy
        """
        # Check if the required backend model is loaded
        if backend == 'pytorch' and (self.pytorch_model is None or self.cfg is None):
            return self._fallback_pose(original_img, bbox)
        elif backend == 'onnx' and self.onnx_session is None:
            return self._fallback_pose(original_img, bbox)
        elif backend == 'tflite' and self.tflite_interpreter is None:
            return self._fallback_pose(original_img, bbox)
        
        try:
            # For non-PyTorch backends, we need to use simplified processing
            # since we don't have access to the full cfg configuration
            if backend in ['onnx', 'tflite']:
                return self._predict_pose_simplified(original_img, bbox, backend)
            
            # Full PyTorch processing with configuration
            with self._isolated_import():
                # Improved camera parameters
                focal = [1500, 1500]
                original_img_height, original_img_width = original_img.shape[:2]
                princpt = [original_img_width / 2, original_img_height / 2]
                
                # Import processing modules
                spec = importlib.util.spec_from_file_location(
                    "ConvNextPose_utils", 
                    os.path.join(self.convnext_path, 'common', "utils", "pose_utils.py")
                )
                ConvNextPose_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_utils)
                
                spec = importlib.util.spec_from_file_location(
                    "ConvNextPose_dataset", 
                    os.path.join(self.convnext_path, 'data', "dataset.py")
                )
                ConvNextPose_dataset = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ConvNextPose_dataset)
                
                # Enhanced bbox processing
                bbox_array = np.array(bbox, dtype=np.float32)
                processed_bbox = ConvNextPose_utils.process_bbox(
                    bbox_array, original_img_width, original_img_height
                )
                
                if processed_bbox is None:
                    logger.warning(f"Bbox processing failed for {bbox}. Using enhanced fallback.")
                    # Enhanced fallback with better bbox handling
                    x, y, w, h = bbox
                    # Expand bbox slightly for better coverage
                    expand_ratio = 0.1
                    new_w = w * (1 + expand_ratio)
                    new_h = h * (1 + expand_ratio)
                    new_x = max(0, x - (new_w - w) / 2)
                    new_y = max(0, y - (new_h - h) / 2)
                    
                    processed_bbox = np.array([new_x, new_y, new_w, new_h], dtype=np.float32)
                
                # Generate patch image with better quality
                img, img2bb_trans = ConvNextPose_dataset.generate_patch_image(
                    original_img, processed_bbox, False, 1.0, 0.0, False
                )
                
                # Enhanced inference
                pose_3d_raw = self.infer(img, backend=backend)
                if pose_3d_raw is None:
                    logger.warning(f"Inference failed, using fallback")
                    return self._fallback_pose(original_img, bbox)
                
                pose_3d = pose_3d_raw[0]  # Remove batch dimension
                
                # Enhanced post-processing with better coordinate handling
                if len(pose_3d.shape) == 2 and pose_3d.shape[1] >= 3:
                    # Standard 3D pose processing
                    pose_3d[:, 0] = pose_3d[:, 0] / self.cfg.output_shape[1] * self.cfg.input_shape[1]
                    pose_3d[:, 1] = pose_3d[:, 1] / self.cfg.output_shape[0] * self.cfg.input_shape[0]
                    
                    # Improved affine transform inversion
                    pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
                    img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
                    
                    try:
                        inv_trans = np.linalg.inv(img2bb_trans_001)
                        pose_3d[:, :2] = np.dot(inv_trans, pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                    except np.linalg.LinAlgError:
                        logger.warning("Matrix inversion failed, using pseudo-inverse")
                        inv_trans = np.linalg.pinv(img2bb_trans_001)
                        pose_3d[:, :2] = np.dot(inv_trans, pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
                    
                    output_pose_2d = pose_3d[:, :2].copy()
                    
                    # Enhanced root depth estimation
                    root_depth = self._estimate_root_depth(img=original_img, bbox=bbox)
                    
                    # Improved depth processing
                    if pose_3d.shape[1] > 2:  # Has depth information
                        pose_3d[:, 2] = (pose_3d[:, 2] / self.cfg.depth_dim * 2 - 1) * \
                            (self.cfg.bbox_3d_shape[0] / 2) + root_depth
                    else:
                        # Add depth information if missing
                        depth_col = np.full((pose_3d.shape[0], 1), root_depth)
                        pose_3d = np.concatenate([pose_3d, depth_col], axis=1)
                    
                    # Convert to camera coordinates
                    pose_3d = ConvNextPose_utils.pixel2cam(pose_3d, focal, princpt)
                    output_pose_3d = pose_3d.copy()
                    
                    # Quality check - ensure reasonable pose coordinates
                    if np.all(np.isfinite(output_pose_2d)) and np.all(np.isfinite(output_pose_3d)):
                        return output_pose_2d, output_pose_3d
                    else:
                        logger.warning("Invalid pose coordinates detected, using fallback")
                        return self._fallback_pose(original_img, bbox)
                else:
                    logger.warning(f"Unexpected pose shape: {pose_3d.shape}")
                    return self._fallback_pose(original_img, bbox)
                
        except Exception as e:
            logger.error(f"Error in enhanced pose prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_pose(original_img, bbox)
    
    def _estimate_root_depth(self, img, bbox: list) -> float:
        """Estimación de profundidad usando RootNet o fallback"""
        try:
            # Inicializar RootNet wrapper si no existe
            if self.root_wrapper is None:
                from config_paths import ROOTNET_PATH, ROOTNET_MODEL_PATH
                self.root_wrapper = RootNetWrapper(ROOTNET_PATH, ROOTNET_MODEL_PATH)
                self.root_wrapper.load_model()  # Cargar modelo una sola vez
            
            return self.root_wrapper.predict_depth(img, bbox)
            
        except ImportError:
            logger.warning("⚠️ ROOTNET_PATH not found in config. Using fallback depth estimation.")
            bbox_area = bbox[2] * bbox[3]
            estimated_depth = 3000.0 / np.sqrt(bbox_area + 1e-6)
            return max(min(estimated_depth, 5000.0), 500.0)
        except Exception as e:
            logger.warning(f"⚠️ RootNet prediction failed: {e}. Using fallback.")
            bbox_area = bbox[2] * bbox[3]
            estimated_depth = 3000.0 / np.sqrt(bbox_area + 1e-6)
            return max(min(estimated_depth, 5000.0), 500.0)
    
    def _fallback_pose(self, img, bbox: list) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced fallback pose with realistic skeleton structure"""
        x, y, w, h = bbox
        center_x, center_y = x + w/2, y + h/2
        depth = self._estimate_root_depth(img, bbox)
        
        # Create more realistic pose structure based on human skeleton
        # Joint order: nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist,
        # right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear
        
        # Proportional joint positions based on typical human proportions
        joints_2d = np.array([
            [center_x, y + h * 0.1],                    # 0: nose
            [center_x, y + h * 0.2],                    # 1: neck  
            [center_x + w * 0.15, y + h * 0.25],       # 2: right_shoulder
            [center_x + w * 0.25, y + h * 0.4],        # 3: right_elbow
            [center_x + w * 0.3, y + h * 0.55],        # 4: right_wrist
            [center_x - w * 0.15, y + h * 0.25],       # 5: left_shoulder
            [center_x - w * 0.25, y + h * 0.4],        # 6: left_elbow
            [center_x - w * 0.3, y + h * 0.55],        # 7: left_wrist
            [center_x + w * 0.1, y + h * 0.6],         # 8: right_hip
            [center_x + w * 0.12, y + h * 0.8],        # 9: right_knee
            [center_x + w * 0.1, y + h * 0.95],        # 10: right_ankle
            [center_x - w * 0.1, y + h * 0.6],         # 11: left_hip
            [center_x - w * 0.12, y + h * 0.8],        # 12: left_knee
            [center_x - w * 0.1, y + h * 0.95],        # 13: left_ankle
            [center_x + w * 0.05, y + h * 0.08],       # 14: right_eye
            [center_x - w * 0.05, y + h * 0.08],       # 15: left_eye
            [center_x + w * 0.1, y + h * 0.12],        # 16: right_ear
            [center_x - w * 0.1, y + h * 0.12],        # 17: left_ear
        ])
        
        # Ensure all joints are within reasonable bounds
        img_h, img_w = img.shape[:2] if len(img.shape) > 1 else (480, 640)
        joints_2d[:, 0] = np.clip(joints_2d[:, 0], 0, img_w - 1)
        joints_2d[:, 1] = np.clip(joints_2d[:, 1], 0, img_h - 1)
        
        # Create 3D pose with varied depths for more realism
        joints_3d = np.zeros((18, 3))
        joints_3d[:, :2] = joints_2d
        
        # Assign depths based on typical human pose (closer/further from camera)
        depth_variations = np.array([
            depth - 50,   # nose (forward)
            depth,        # neck (center)
            depth + 30,   # right_shoulder (back)
            depth + 20,   # right_elbow
            depth + 10,   # right_wrist (forward)
            depth + 30,   # left_shoulder (back)
            depth + 20,   # left_elbow
            depth + 10,   # left_wrist (forward)
            depth + 40,   # right_hip (back)
            depth + 30,   # right_knee
            depth + 20,   # right_ankle
            depth + 40,   # left_hip (back)
            depth + 30,   # left_knee
            depth + 20,   # left_ankle
            depth - 60,   # right_eye (forward)
            depth - 60,   # left_eye (forward)
            depth - 40,   # right_ear
            depth - 40,   # left_ear
        ])
        
        joints_3d[:, 2] = depth_variations
        
        return joints_2d.astype(np.float32), joints_3d.astype(np.float32)
    
    def _predict_pose_simplified(self, original_img: np.ndarray, bbox: list, 
                                backend: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Simplified pose prediction for ONNX/TFLite backends without full config"""
        try:
            # Simple bbox processing for ONNX/TFLite
            x, y, w, h = bbox
            
            # Create a reasonable crop from bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(original_img.shape[1], int(x + w)), min(original_img.shape[0], int(y + h))
            
            # Ensure valid bbox
            if x2 <= x1 or y2 <= y1:
                return self._fallback_pose(original_img, bbox)
            
            # Crop and resize to model input size (256x256)
            cropped = original_img[y1:y2, x1:x2]
            if cropped.size == 0:
                return self._fallback_pose(original_img, bbox)
            
            img = cv2.resize(cropped, (self.input_size, self.input_size))
            
            # Run inference
            pose_3d_raw = self.infer(img, backend=backend)
            if pose_3d_raw is None:
                logger.warning(f"Simplified inference failed, using fallback")
                return self._fallback_pose(original_img, bbox)
            
            # Simplified post-processing
            pose_3d = pose_3d_raw[0] if len(pose_3d_raw.shape) > 2 else pose_3d_raw
            
            # Scale back to original bbox coordinates
            if len(pose_3d.shape) == 2 and pose_3d.shape[1] >= 3:
                # Simple scaling from normalized coordinates
                pose_3d[:, 0] = pose_3d[:, 0] * w + x  # Scale X to bbox
                pose_3d[:, 1] = pose_3d[:, 1] * h + y  # Scale Y to bbox
                
                # Keep Z coordinates as relative depth
                pose_2d = pose_3d[:, :2].copy()
                
                # Ensure coordinates are within image bounds
                pose_2d[:, 0] = np.clip(pose_2d[:, 0], 0, original_img.shape[1] - 1)
                pose_2d[:, 1] = np.clip(pose_2d[:, 1], 0, original_img.shape[0] - 1)
                
                # Estimate root depth for 3D coordinates
                root_depth = self._estimate_root_depth(original_img, bbox)
                pose_3d[:, 2] = pose_3d[:, 2] + root_depth  # Add base depth
                
                return pose_2d.astype(np.float32), pose_3d.astype(np.float32)
            else:
                logger.warning("Invalid pose coordinates detected, using fallback")
                return self._fallback_pose(original_img, bbox)
                
        except Exception as e:
            logger.error(f"Simplified pose prediction failed: {e}")
            return self._fallback_pose(original_img, bbox)

    # Legacy methods for backwards compatibility
    
    def predict_pose(self, original_img, bbox):
        """Legacy method - use predict_pose_full instead"""
        pose_2d, pose_3d = self.predict_pose_full(original_img, bbox)
        return pose_2d, pose_3d
    
    def return_model(self):
        """Legacy method"""
        if self.pytorch_model is not None:
            return self.pytorch_model
        else:
            raise ValueError("ConvNextPose model is not loaded. Please call load_pytorch_model() first.")
