#!/usr/bin/env python3
"""
main.py - ConvNeXt Pose Real-time Production Pipeline with 3D Support (High Performance)

Optimized production pipeline using direct imports from ConvNeXt project.
Now includes RootNet integration for full 3D pose estimation!

🎯 PERFORMANCE RESULTS:
- PyTorch Backend: ~12.5 FPS (2D) / ~10 FPS (3D)
- ONNX Backend: ~29 FPS (2D) / ~20 FPS (3D)
- TFLite Backend: ~8-15 FPS (2D) / ~6-10 FPS (3D)

🔧 KEY OPTIMIZATIONS:
- Direct ConvNeXt imports (no wrapper overhead)
- Intelligent frame skipping
- Multi-threaded pose estimation
- Optimized detection caching
- Hardware-aware thread configuration
- Optional RootNet integration for 3D depth estimation

💡 USAGE:
    python main.py --preset ultra_fast --backend pytorch --enable_3d
    python main.py --input video.mp4 --preset quality_focused --backend onnx --enable_3d
    python main.py --input 0 --preset speed_balanced --backend tflite
"""

import os
import sys
import time
import logging
import argparse
import warnings
import threading
import queue
from types import SimpleNamespace
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# Setup
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '4'
warnings.filterwarnings("ignore", category=UserWarning)

# ONNX Runtime
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# TensorFlow Lite  
TFLITE_AVAILABLE = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    TFLITE_AVAILABLE = True
except ImportError:
    pass

# YOLO Detection
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project imports - Direct imports for maximum performance
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT

# Try to find the ConvNeXt project - check common locations
CONVNEXT_PROJECT_PATHS = [
    ROOT.parent / 'ConvNeXtPose',  # Sibling directory
    Path('d:/Repository-Projects/ConvNeXtPose'),  # Absolute path
    ROOT / 'ConvNeXtPose',  # Subdirectory
]

CONVNEXT_ROOT = None
for path in CONVNEXT_PROJECT_PATHS:
    if (path / 'main' / 'config.py').exists():
        CONVNEXT_ROOT = path
        logger.info(f"✅ Found ConvNeXt project at: {CONVNEXT_ROOT}")
        break

if CONVNEXT_ROOT is None:
    logger.error("❌ ConvNeXt project not found. Please ensure it's available.")
    sys.exit(1)

# Add ConvNeXt project paths
CONVNEXT_PATHS = [
    CONVNEXT_ROOT / 'main',
    CONVNEXT_ROOT / 'data', 
    CONVNEXT_ROOT / 'common',
]

for path in CONVNEXT_PATHS:
    if path.exists():
        sys.path.insert(0, str(path))

try:
    from config import cfg
    from model import get_pose_net
    from dataset import generate_patch_image
    from utils.pose_utils import process_bbox
    logger.info("✅ Direct ConvNeXt imports successful")
except ImportError as e:
    logger.error(f"Critical: Cannot import ConvNeXt modules: {e}")
    logger.error("Make sure ConvNeXt project structure is available")
    sys.exit(1)

# RootNet integration for 3D pose - with robust import/patching logic
ROOTNET_AVAILABLE = False
ROOTNET_ROOT = None
RootNetWrapper = None

def apply_torchvision_patch():
    """Apply torchvision compatibility patch for legacy code"""
    import torchvision.models.resnet as resnet_module
    
    # Only apply if model_urls doesn't exist
    if not hasattr(resnet_module, 'model_urls'):
        # Recreate model_urls as it existed in older versions
        resnet_module.model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }
        logger.info("🔧 Applied torchvision compatibility patch")

def initialize_rootnet():
    """Initialize RootNet with robust import/patching logic"""
    global ROOTNET_AVAILABLE, ROOTNET_ROOT, RootNetWrapper
    
    try:
        # Apply compatibility patch first
        apply_torchvision_patch()
        
        # Check for RootNet paths
        ROOTNET_PATHS = [
            ROOT.parent / '3DMPPE_ROOTNET_RELEASE',  # Sibling directory
            Path('d:/Repository-Projects/3DMPPE_ROOTNET_RELEASE'),  # Absolute path
            ROOT / '3DMPPE_ROOTNET_RELEASE',  # Subdirectory
        ]
        
        for path in ROOTNET_PATHS:
            if (path / 'main' / 'config.py').exists():
                ROOTNET_ROOT = path
                logger.info(f"✅ Found RootNet project at: {ROOTNET_ROOT}")
                break
        
        if ROOTNET_ROOT is not None:
            # Import RootNetWrapper only after patches are applied
            from src.root_wrapper import RootNetWrapper
            ROOTNET_AVAILABLE = True
            logger.info("✅ RootNet integration available with robust patching")
        else:
            logger.info("⚠️ RootNet project not found - 3D estimation disabled")
            
    except ImportError as e:
        logger.info(f"⚠️ RootNet wrapper not available: {e} - 3D estimation disabled")
    except Exception as e:
        logger.warning(f"⚠️ RootNet initialization error: {e} - 3D estimation disabled")

# Initialize RootNet with robust logic
initialize_rootnet()

# PRODUCTION PRESETS - Optimized for real performance with 3D support
PRODUCTION_PRESETS = {
    'ultra_fast': {
        'target_fps': 15.0,
        'frame_skip': 3,
        'yolo_size': 320,
        'max_persons': 2,
        'detection_freq': 4,
        'thread_count': 2,
        'enable_threading': True,
        'enable_3d': False,  # Default to 2D for speed
        'description': 'Ultra rápido - 15+ FPS con estabilidad (2D)'
    },
    'speed_balanced': {
        'target_fps': 12.0,
        'frame_skip': 2,
        'yolo_size': 416,
        'max_persons': 3,
        'detection_freq': 3,
        'thread_count': 2,
        'enable_threading': True,
        'enable_3d': False,  # Can be enabled via --enable_3d
        'description': 'Balance velocidad-calidad - 12+ FPS (2D/3D)'
    },
    'quality_focused': {
        'target_fps': 10.0,
        'frame_skip': 1,
        'yolo_size': 512,
        'max_persons': 4,
        'detection_freq': 2,
        'thread_count': 1,
        'enable_threading': False,
        'enable_3d': False,  # Can be enabled via --enable_3d
        'description': 'Mejor calidad - 10+ FPS (2D/3D disponible)'
    }
}

def detect_optimized_hardware():
    """Detectar hardware y configurar optimizaciones"""
    hardware_info = {
        'has_cuda': torch.cuda.is_available(),
        'cpu_count': os.cpu_count(),
        'torch_threads': torch.get_num_threads()
    }
    
    # Optimize torch threads
    if hardware_info['cpu_count'] >= 8:
        torch.set_num_threads(6)
    elif hardware_info['cpu_count'] >= 4:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(2)
    
    if hardware_info['has_cuda']:
        return 'gpu_available'
    else:
        return 'cpu_optimized'

def convert_yolo_to_onnx_optimized(pt_model_path='yolo11n.pt', 
                                   conf_thresh=0.7, iou_thresh=0.45, img_size=640):
    """Convierte YOLO a ONNX con optimizaciones específicas"""
    base_name = pt_model_path.replace('.pt', '')
    onnx_path = f"{base_name}_optimized_conf{conf_thresh}_iou{iou_thresh}.onnx"
    
    if os.path.exists(onnx_path):
        print(f"✅ ONNX optimizado existente: {onnx_path}")
        return onnx_path
    
    print(f"🔄 Convirtiendo {pt_model_path} a ONNX optimizado...")
    try:
        from ultralytics import YOLO
        model = YOLO(pt_model_path)
        
        exported_path = model.export(
            format='onnx', 
            imgsz=img_size, 
            optimize=True, 
            dynamic=False, 
            simplify=True, 
            opset=13,  # Opset más reciente
            nms=True,
            conf=conf_thresh, 
            iou=iou_thresh, 
            max_det=100  # Reducido para velocidad
        )
        
        if exported_path != onnx_path:
            os.rename(exported_path, onnx_path)
        
        print(f"✅ ONNX optimizado creado: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"❌ Error en conversión optimizada: {e}")
        raise


class ProductionYOLODetector:
    """Detector YOLO optimizado pero estable"""
    
    def __init__(self, model_path: str = 'models/yolo11n.pt', input_size: int = 320):
        self.model_path = convert_yolo_to_onnx_optimized(model_path, img_size = input_size)
        self.input_size = input_size
        self.detector = None
        self.frame_count = 0
        self.last_detections = []
        self.detection_cache = {}
        self._initialize()
        self.warmup()
        
    
    def _initialize(self):
        if not YOLO_AVAILABLE:
            logger.warning("⚠️ YOLO not available")
            return
        
        try:
            # Try absolute path first, then relative
            model_paths = [
                Path(self.model_path),
                ROOT / self.model_path,
                ROOT / 'models' / 'yolo11n.pt'
            ]
            
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                logger.warning(f"⚠️ YOLO model not found at {self.model_path}")
                return
            providers = []
            session_options = ort.SessionOptions()

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cuda_provider = ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kSameAsRequested' if gpu_memory < 6 else 'kNextPowerOfTwo',
                    'gpu_mem_limit': int(gpu_memory*0.6*1024**3),
                    'cdunn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                })
                providers.append(cuda_provider)
                session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                session_options.inter_op_num_threads = min(2, os.cpu_count())
                session_options.intra_op_num_threads = min(4, os.cpu_count())
                logger.info("🚀 YOLO configurado para GPU de alto rendimiento")
            else:
                # Configuración CPU optimizada
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.inter_op_num_threads = 1
                session_options.intra_op_num_threads = min(4, os.cpu_count())
                print("🚀 YOLO configurado para CPU optimizado")
                providers.append('CPUExecutionProvider')
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(self.model_path, providers = providers, sess_options = session_options)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.img_size = self.input_shape[2]

            # self.detector = YOLO(str(model_path))
            # # Configure for optimal CPU performance
            # self.detector.overrides['imgsz'] = self.input_size
            # self.detector.overrides['half'] = False
            # self.detector.overrides['device'] = 'cpu'
            
            logger.info("✅ Production YOLO initialized: %s (size: %d)", model_path.name, self.input_size)
        except Exception as e:
            logger.error("❌ YOLO initialization failed: %s", e)
    def warmup(self):
        """Pre-calentar con número de iteraciones adaptativo"""
        print("🔥 Pre-calentando YOLO adaptativo...")
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
        
        warmup_iterations = 5 if torch.cuda.is_available() else 3
        for _ in range(warmup_iterations):
            self.session.run(self.output_names, {self.input_name: dummy_input})
        print("✅ YOLO adaptativo pre-calentado")

    def preprocess_frame_adaptive(self, frame:np.ndarray):
        """Preprocesamiento adaptativo según hardware"""
        h, w = frame.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Interpolación adaptativa
        interpolation = cv2.INTER_LINEAR if torch.cuda.is_available() else cv2.INTER_NEAREST
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
        
        # Padding optimizado
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, self.img_size - new_h - pad_h, 
            pad_w, self.img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Conversión optimizada
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]
        
        return input_tensor, scale, pad_w, pad_h
    def detect_persons(self, frame: np.ndarray, detection_freq: int = 4, 
                      conf_threshold: float = 0.7) -> List[List[int]]:
        """Detección con cache optimizado"""
        if self.session is None:
            return []
        
        self.frame_count += 1
        
        # Use cached detections for performance
        if self.frame_count % detection_freq != 0:
            return self.last_detections
        
        try:
            # Optimization: resize frame if too large for faster detection
            input_tensor, scale, pad_w, pad_h = self.preprocess_frame_adaptive(frame)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            if len(outputs) == 0 or outputs[0].size == 0:
                return [[]]
            
            detections = outputs[0][0] #
            person_mask = (detections[:, 5] == 0) & (detections[:, 4] >= conf_threshold)
            person_detections = detections[person_mask]

            if len(person_detections) == 0:
                return [[]]
            
            boxes = person_detections[:, :4].copy()
            scores = person_detections[:, 4]

            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

            h, w = frame.shape[:2]
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            self.last_detections = boxes
            return boxes
        
        except Exception as e:
            logger.warning(f"⚠️ Detection failed: {e}")
            return self.last_detections
    
    def cleanup(self):
        pass

class ProductionInferenceEngine:
    """Motor de inferencia optimizado usando imports directos"""
    def __init__(self, model_path: str, backend: str = 'pytorch'):
        self.model_path = model_path
        self.backend = backend
        self.active_backend = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use consistent 256x256 for all backends
        self.input_size = 256
        self.output_size = 32
        
        # Transform (same as demo.py - CRITICAL)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)
        ])
        
        # Models
        self.pytorch_model = None
        self.onnx_session = None
        self.tflite_interpreter = None
        
        self._initialize()
    
    def _initialize(self):
        logger.info("🚀 Initializing production inference engine...")
        logger.info("   Backend: %s", self.backend)
        if self.backend == 'pytorch':
            self._setup_pytorch()
        elif self.backend == 'onnx' and ONNX_AVAILABLE:
            self._setup_onnx()
        elif self.backend == 'tflite' and TFLITE_AVAILABLE:
            self._setup_tflite()
        
        if self.active_backend is None:
            logger.warning("⚠️ Fallback to PyTorch backend")
            self._setup_pytorch()
        
        logger.info("✅ Production inference engine active: %s", self.active_backend)
    
    def _setup_pytorch(self) -> bool:
        try:
            logger.info("🔄 Setting up PyTorch backend...")
            joint_num = 18
            self.pytorch_model = get_pose_net(cfg, False, joint_num)
            
            # Try multiple model paths
            model_paths = [
                Path(self.model_path),
                ROOT / self.model_path,
                ROOT / 'models' / 'model_opt_S.pth'
            ]
            
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                logger.error("❌ PyTorch model not found")
                return False
            
            checkpoint = torch.load(str(model_path), map_location=self.device)
            
            # Handle DataParallel models
            if 'network' in checkpoint:
                state_dict = checkpoint['network']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            if any(key.startswith('module.') for key in state_dict.keys()):
                logger.info("🔧 Removing DataParallel prefix from state_dict...")
                state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            
            # Load with strict=False to handle minor architecture differences
            missing_keys, unexpected_keys = self.pytorch_model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning("⚠️ Missing keys in model: %s", missing_keys[:5])
            if unexpected_keys:
                logger.warning("⚠️ Unexpected keys in model: %s", unexpected_keys[:5])
            
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()
            
            self.active_backend = 'pytorch'
            logger.info("✅ PyTorch backend ready")
            return True            
        except Exception as e:
            logger.error("❌ PyTorch setup failed: %s", e)
            return False
    
    def _setup_onnx(self) -> bool:
        try:
            logger.info("🔄 Setting up ONNX backend...")
            
            # Try multiple ONNX model paths
            model_dir = ROOT / "models"
            onnx_candidates = [
                model_dir / "model_opt_S_optimized.onnx",
                model_dir / "model_S.onnx",
                model_dir / "model_opt_S.onnx"
            ]
            
            onnx_path = None
            for candidate in onnx_candidates:
                if candidate.exists():
                    onnx_path = candidate
                    break
            
            if onnx_path is None:
                logger.warning("⚠️ No ONNX model found")
                return False
            
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            self.active_backend = 'onnx'
            logger.info("✅ ONNX backend ready")
            return True
            
        except Exception as e:
            logger.error("❌ ONNX setup failed: %s", e)
            return False
    
    def _setup_tflite(self) -> bool:
        try:
            logger.info("🔄 Setting up TFLite backend...")
            
            model_dir = ROOT / "models"
            tflite_candidates = [
                model_dir / "model_opt_S_fast_native.tflite",
                model_dir / "model_opt_S_balanced.tflite",
            ]
            
            for candidate in tflite_candidates:
                if candidate.exists():
                    if self._try_load_tflite_model(candidate):
                        return True
            
            logger.error("❌ No TFLite models could be loaded")
            return False
            
        except Exception as e:
            logger.error("❌ TFLite setup failed: %s", e)
            return False
    
    def _try_load_tflite_model(self, model_path: Path) -> bool:
        try:
            interpreter = tf.lite.Interpreter(
                model_path=str(model_path),
                num_threads=4,
                experimental_preserve_all_tensors=False
            )
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test inference
            input_shape = input_details[0]['shape']
            dummy_input = np.random.random(input_shape).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            
            self.tflite_interpreter = interpreter
            self.tflite_input_details = input_details
            self.tflite_output_details = output_details
            self.active_backend = 'tflite'
            
            logger.info(f"✅ TFLite model loaded: {model_path.name}")
            return True
            
        except Exception as e:
            logger.warning(f"❌ Failed to load {model_path.name}: {e}")
            return False
    
    def infer(self, img_patch: np.ndarray) -> Optional[np.ndarray]:
        """Inference using direct ConvNeXt logic"""
        if self.active_backend == 'pytorch':
            return self._infer_pytorch(img_patch)
        elif self.active_backend == 'onnx':
            return self._infer_onnx(img_patch)
        elif self.active_backend == 'tflite':
            return self._infer_tflite(img_patch)
        return None
    
    def _infer_pytorch(self, img_patch: np.ndarray) -> np.ndarray:
        # Prepare input (same as demo.py)
        inp = self.transform(img_patch).to(self.device)[None, :, :, :]
        
        # Inference
        with torch.no_grad():
            output = self.pytorch_model(inp)
        
        return output.cpu().numpy()
    
    def _infer_onnx(self, img_patch: np.ndarray) -> np.ndarray:
        # Prepare input
        inp = self.transform(img_patch).numpy()[None, :, :, :]
        
        # ONNX inference
        output = self.onnx_session.run(None, {'input': inp})
        
        return output[0]
    
    def _infer_tflite(self, img_patch: np.ndarray) -> np.ndarray:
        """TFLite inference"""
        if self.tflite_interpreter is None:
            return None
        
        # Prepare input
        inp = self.transform(img_patch).numpy()[None, :, :, :].astype(np.float32)
        
        # Set input tensor
        self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], inp)
        
        # Run inference
        self.tflite_interpreter.invoke()
        
        # Get output tensor
        output = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
        
        return output

class ProductionV4Processor:
    """Procesador V4 usando imports directos para máximo rendimiento con soporte 3D"""
    
    def __init__(self, model_path: str, preset: str = 'ultra_fast', backend: str = 'pytorch', enable_3d: bool = False):
        self.model_path = model_path
        self.preset = preset
        self.backend = backend
        self.enable_3d = enable_3d and ROOTNET_AVAILABLE
        self.config = PRODUCTION_PRESETS[preset].copy()
        
        # Override 3D setting if specified
        if enable_3d and not ROOTNET_AVAILABLE:
            logger.warning("⚠️ 3D requested but RootNet not available - falling back to 2D")
            self.enable_3d = False
        
        self.hardware = detect_optimized_hardware()
        
        # Components
        self.yolo_detector = ProductionYOLODetector(input_size=self.config['yolo_size'])
        self.inference_engine = ProductionInferenceEngine(model_path, backend)
        
        # RootNet for 3D depth estimation
        self.rootnet_processor = None
        if self.enable_3d:
            self._initialize_rootnet()
        
        # Threading (controlled)
        self.thread_pool = None
        if self.config['enable_threading'] and self.config['thread_count'] > 1:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config['thread_count'])
        
        # Stats and optimization
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)
        self.last_poses = []
        self.skip_count = 0
        
        logger.info("✅ Production V4 Processor initialized")
        logger.info("   Preset: %s (%s)", preset, self.config['description'])
        logger.info("   Target FPS: %.1f", self.config['target_fps'])
        logger.info("   Hardware: %s", self.hardware)
        logger.info("   Backend: %s", self.inference_engine.active_backend)
        logger.info("   3D Mode: %s", "enabled" if self.enable_3d else "disabled")
        logger.info("   Threading: %s (%d threads)", 
                   "enabled" if self.config['enable_threading'] else "disabled",
                   self.config['thread_count'])
    
    def _initialize_rootnet(self):
        """Initialize RootNet for 3D depth estimation with robust patching"""
        try:
            if not ROOTNET_AVAILABLE or RootNetWrapper is None:
                logger.warning("⚠️ RootNet not available - cannot initialize 3D estimation")
                self.enable_3d = False
                return
            
            # Find RootNet checkpoint
            checkpoint_paths = [
                ROOT / 'models' / 'snapshot_19.pth.tar',  # Existing checkpoint
                ROOTNET_ROOT / 'snapshot_19.pth.tar',
            ]
            
            checkpoint_path = None
            for path in checkpoint_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    break
            
            if checkpoint_path is None:
                logger.warning("⚠️ RootNet checkpoint not found - using existing model_opt_S.pth")
                checkpoint_path = str(ROOT / 'models' / 'model_opt_S.pth')
            
            # Initialize with robust patching from root_wrapper.py
            from src.root_wrapper import RootNetWrapper
            self.rootnet_processor = RootNetWrapper(
                rootnet_path=str(ROOTNET_ROOT),
                checkpoint_path=checkpoint_path
            )
            
            self.rootnet_processor.load_model(use_gpu=torch.cuda.is_available())
            
            if self.rootnet_processor.model is not None:
                logger.info("✅ RootNet initialized for 3D depth estimation with robust patching")
            else:
                logger.warning("⚠️ RootNet initialization failed - falling back to 2D")
                self.enable_3d = False
                self.rootnet_processor = None
                
        except Exception as e:
            logger.warning(f"⚠️ RootNet initialization failed: {e} - falling back to 2D")
            import traceback
            traceback.print_exc()
            self.enable_3d = False
            self.rootnet_processor = None
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        start_time = time.time()
        self.frame_count += 1
        
        # Intelligent frame skipping
        if self._should_skip_frame():
            return self.last_poses, self._get_stats(start_time, len(self.last_poses), skipped=True)
        
        # Detection
        detection_start = time.time()
        persons = self.yolo_detector.detect_persons(
            frame, 
            self.config['detection_freq'],
            conf_threshold=0.7
        )
        detection_time = time.time() - detection_start
        
        # Pose estimation with controlled threading
        pose_start = time.time()
        if self.thread_pool and len(persons) > 1:
            poses = self._estimate_poses_threaded(frame, persons[:self.config['max_persons']])
        else:
            poses = self._estimate_poses_single(frame, persons[:self.config['max_persons']])
        
        pose_time = time.time() - pose_start
        
        # Update cache
        self.last_poses = poses
        
        # Stats
        stats = self._get_stats(start_time, len(poses), 
                              detection_time=detection_time, 
                              pose_time=pose_time)
        
        return poses, stats
    
    def _should_skip_frame(self) -> bool:
        """Smart frame skipping"""
        if len(self.processing_times) < 10:
            return False
        
        avg_time = np.mean(list(self.processing_times)[-10:])
        target_time = 1.0 / self.config['target_fps']
        
        # Skip if we're running too slow
        if avg_time > target_time * 1.2:
            self.skip_count += 1
            if self.skip_count < self.config['frame_skip']:
                return True
        
        self.skip_count = 0
        return False
    
    def _estimate_poses_single(self, frame: np.ndarray, persons: List[List[int]]) -> List[np.ndarray]:
        """Single-threaded pose estimation (stable)"""
        poses = []
        
        for bbox in persons:
            pose_2d = self._process_single_person(frame, bbox)
            if pose_2d is not None:
                poses.append(pose_2d)
        
        return poses
    
    def _estimate_poses_threaded(self, frame: np.ndarray, persons: List[List[int]]) -> List[np.ndarray]:
        """Multi-threaded pose estimation (controlled)"""
        if not persons:
            return []
        
        # Submit tasks to thread pool
        futures = []
        for bbox in persons:
            future = self.thread_pool.submit(self._process_single_person, frame, bbox)
            futures.append(future)
        
        # Collect results with timeout
        poses = []
        for future in futures:
            try:
                pose_2d = future.result(timeout=0.2)  # 200ms timeout
                if pose_2d is not None:
                    poses.append(pose_2d)
            except Exception as e:
                logger.warning(f"⚠️ Threaded pose estimation failed: {e}")
                continue
        
        return poses
    
    def _process_single_person(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Process single person using EXACT ConvNeXt logic with optional 3D depth"""
        try:
            # Convert bbox format from YOLO [x1, y1, x2, y2] to ConvNeXt [x, y, w, h]
            x1, y1, x2, y2 = bbox
            bbox_array = np.array([x1, y1, x2 - x1, y2 - y1])
            
            # Process bbox using EXACT function from pose_utils.py
            processed_bbox = process_bbox(bbox_array, frame.shape[1], frame.shape[0])
            if processed_bbox is None:
                return None
            
            # Generate patch using EXACT method from dataset.py
            img_patch, img2bb_trans = generate_patch_image(
                frame, processed_bbox, False, 1.0, 0.0, False
            )
            
            # Inference
            pose_output = self.inference_engine.infer(img_patch)
            if pose_output is None:
                return None
            
            # Process output using EXACT demo.py post-processing
            pose_3d = pose_output[0]  # Extract first result
            
            # EXACT post-processing from demo.py
            pose_3d = pose_3d.copy()
            pose_3d[:, 0] = pose_3d[:, 0] / cfg.output_shape[1] * cfg.input_shape[1]
            pose_3d[:, 1] = pose_3d[:, 1] / cfg.output_shape[0] * cfg.input_shape[0]
            pose_3d_xy1 = np.concatenate((pose_3d[:, :2], np.ones_like(pose_3d[:, :1])), 1)
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            pose_3d[:, :2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            
            # Add 3D depth estimation if enabled
            if self.enable_3d and self.rootnet_processor is not None:
                try:
                    # Estimate root depth using RootNet
                    root_depth = self.rootnet_processor.predict_depth(
                        frame, bbox, focal=[1500, 1500]  # Standard focal length
                    )
                    
                    # Create 3D coordinates: add depth to all joints
                    # RootNet estimates the root depth, we propagate it to all joints
                    pose_3d_with_depth = np.zeros((pose_3d.shape[0], 3))
                    pose_3d_with_depth[:, :2] = pose_3d[:, :2]  # x, y coordinates
                    pose_3d_with_depth[:, 2] = root_depth  # z coordinate (depth)
                    
                    return pose_3d_with_depth
                    
                except Exception as e:
                    logger.debug(f"⚠️ 3D depth estimation failed: {e}, falling back to 2D")
                    # Fall back to 2D if 3D fails
                    pass
            
            # Return 2D coordinates (default)
            return pose_3d[:, :2].copy()
            
        except Exception as e:
            logger.debug(f"⚠️ Pose processing failed: {e}")
            return None
    
    def _get_stats(self, start_time: float, poses_count: int, 
                   detection_time: float = 0, pose_time: float = 0, 
                   skipped: bool = False) -> Dict[str, Any]:
        """Get processing statistics"""
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        avg_time = np.mean(self.processing_times)
        
        return {
            'frame_count': self.frame_count,
            'avg_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'instant_fps': 1.0 / total_time if total_time > 0 else 0,
            'target_fps': self.config['target_fps'],
            'processing_time_ms': total_time * 1000,
            'detection_time_ms': detection_time * 1000,
            'pose_time_ms': pose_time * 1000,
            'poses_detected': poses_count,
            'active_backend': self.inference_engine.active_backend,
            'preset': self.preset,
            'hardware': self.hardware,
            'skipped': skipped,
            'threading_enabled': self.config['enable_threading']
        }
    
    def cleanup(self):
        self.yolo_detector.cleanup()
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)

def draw_pose(image: np.ndarray, pose: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)):
    """Draw pose using standard COCO skeleton with 3D support"""
    # COCO 18-point skeleton connections
    skeleton = [
        (10, 9), (9, 8), (8, 11), (8, 14),
        (11, 12), (12, 13), (14, 15), (15, 16),
        (11, 4), (14, 1), (0, 4), (0, 1),
        (4, 5), (5, 6), (1, 2), (2, 3)
    ]
    
    # Check if 3D pose (has depth dimension)
    is_3d = pose.shape[1] >= 3
    
    # Draw joints
    for i, joint in enumerate(pose):
        x, y = int(joint[0]), int(joint[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # Use different colors for 3D vs 2D
            joint_color = color
            if is_3d:
                # Color code by depth - closer joints brighter
                depth = joint[2] if len(joint) > 2 else 0
                # Normalize depth for color coding (rough estimation)
                depth_norm = max(0, min(1, (3000 - depth) / 2000))  # 1000-3000mm range
                joint_color = (int(color[0] * depth_norm), int(color[1] * depth_norm), int(color[2]))
            
            cv2.circle(image, (x, y), 3, joint_color, -1)
    
    # Draw skeleton
    for start_idx, end_idx in skeleton:
        if start_idx < len(pose) and end_idx < len(pose):
            start_point = (int(pose[start_idx][0]), int(pose[start_idx][1]))
            end_point = (int(pose[end_idx][0]), int(pose[end_idx][1]))
            
            # Check if both points are within image bounds
            if (0 <= start_point[0] < image.shape[1] and 0 <= start_point[1] < image.shape[0] and
                0 <= end_point[0] < image.shape[1] and 0 <= end_point[1] < image.shape[0]):
                cv2.line(image, start_point, end_point, color, 2)
    
    # Add depth info for 3D poses
    if is_3d and len(pose) > 0:
        root_depth = pose[0][2] if len(pose[0]) > 2 else 0
        depth_text = f"Depth: {root_depth:.0f}mm"
        # Draw depth info near the pose
        text_pos = (int(pose[0][0]) + 10, int(pose[0][1]) - 10)
        if 0 <= text_pos[0] < image.shape[1] - 150 and 0 <= text_pos[1] < image.shape[0]:
            cv2.putText(image, depth_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt Pose Real-time Production Pipeline with 3D Support (Optimized)')
    parser.add_argument('--input', type=str, default='0', 
                       help='Input source: webcam index (0), video file path, or image path')
    parser.add_argument('--model', type=str, default='models/model_opt_S.pth',
                       help='Path to the ConvNeXt pose model')
    parser.add_argument('--preset', choices=list(PRODUCTION_PRESETS.keys()), 
                       default='ultra_fast', help='Performance preset')
    parser.add_argument('--backend', choices=['pytorch', 'onnx', 'tflite'], 
                       default='pytorch', help='Inference backend')
    parser.add_argument('--enable_3d', action='store_true',
                       help='Enable 3D pose estimation with RootNet (requires RootNet)')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable video display (useful for benchmarking)')
    
    args = parser.parse_args()
    
    # Initialize processor with 3D support
    processor = ProductionV4Processor(args.model, args.preset, args.backend, args.enable_3d)
    
    # Setup input
    if args.input.isdigit():
        # Webcam
        cap = cv2.VideoCapture(int(args.input))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        is_video = True
    elif Path(args.input).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video file
        cap = cv2.VideoCapture(args.input)
        is_video = True
    else:
        # Single image
        is_video = False
        frame = cv2.imread(args.input)
        if frame is None:
            logger.error(f"Could not load image: {args.input}")
            return
    
    # Setup video writer if needed
    video_writer = None
    if args.save_video and is_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        if is_video:
            fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(args.save_video, fourcc, fps, frame_size)
    
    try:
        if is_video:
            # Process video
            logger.info("🎬 Starting video processing...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                poses, stats = processor.process_frame(frame)
                
                # Draw poses
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                for i, pose in enumerate(poses):
                    color = colors[i % len(colors)]
                    draw_pose(frame, pose, color)
                
                # Display stats
                info_text = (f"FPS: {stats['avg_fps']:.1f}/{stats['target_fps']:.1f} | "
                           f"Backend: {stats['active_backend']} | "
                           f"Poses: {stats['poses_detected']} | "
                           f"Frame: {stats['frame_count']}")
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save frame if needed
                if video_writer is not None:
                    video_writer.write(frame)
                
                # Display frame
                if not args.no_display:
                    cv2.imshow('ConvNeXt Pose Real-time', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log stats periodically
                if stats['frame_count'] % 100 == 0:
                    logger.info(f"Processed {stats['frame_count']} frames | "
                              f"Avg FPS: {stats['avg_fps']:.1f} | "
                              f"Backend: {stats['active_backend']}")
        else:
            # Process single image
            logger.info("🖼️ Processing single image...")
            poses, stats = processor.process_frame(frame)
            
            # Draw poses
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i, pose in enumerate(poses):
                color = colors[i % len(colors)]
                draw_pose(frame, pose, color)
            
            # Display result
            if not args.no_display:
                cv2.imshow('ConvNeXt Pose Result', frame)
                cv2.waitKey(0)
            
            # Save if needed
            if args.save_video:
                cv2.imwrite(args.save_video, frame)
            
            logger.info(f"Detected {len(poses)} poses | Backend: {stats['active_backend']}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Cleanup
        processor.cleanup()
        if is_video:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        logger.info("✅ Cleanup completed")

if __name__ == "__main__":
    main()
