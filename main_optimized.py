#!/usr/bin/env python3
"""
main_optimized.py - ConvNeXt Pose Real-time Production Pipeline (Maximum Performance)

High-performance production pipeline using direct imports from the original ConvNeXt project.
Based on the proven convnext_realtime_v4_production_optimized.py architecture.

Usage:
    python main_optimized.py --preset ultra_fast --backend pytorch
    python main_optimized.py --input video.mp4 --preset quality_focused --backend onnx
    python main_optimized.py --input 0 --preset speed_balanced --backend tflite
"""

import os
import sys
import time
import logging
import argparse
import warnings
import threading
import queue
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

# PRODUCTION PRESETS - Optimized for real performance
PRODUCTION_PRESETS = {
    'ultra_fast': {
        'target_fps': 15.0,
        'frame_skip': 3,
        'yolo_size': 320,
        'max_persons': 2,
        'detection_freq': 4,
        'thread_count': 2,
        'enable_threading': True,
        'description': 'Ultra rápido - 15+ FPS con estabilidad'
    },
    'speed_balanced': {
        'target_fps': 12.0,
        'frame_skip': 2,
        'yolo_size': 416,
        'max_persons': 3,
        'detection_freq': 3,
        'thread_count': 2,
        'enable_threading': True,
        'description': 'Balance velocidad-calidad - 12+ FPS'
    },
    'quality_focused': {
        'target_fps': 10.0,
        'frame_skip': 1,
        'yolo_size': 512,
        'max_persons': 4,
        'detection_freq': 2,
        'thread_count': 1,
        'enable_threading': False,
        'description': 'Mejor calidad - 10+ FPS sin threading'
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

class ProductionYOLODetector:
    """Detector YOLO optimizado pero estable"""
    
    def __init__(self, model_path: str = 'models/yolo11n.pt', input_size: int = 320):
        self.model_path = model_path
        self.input_size = input_size
        self.detector = None
        self.frame_count = 0
        self.last_detections = []
        self.detection_cache = {}
        
        self._initialize()
    
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
            
            self.detector = YOLO(str(model_path))
            # Configure for optimal CPU performance
            self.detector.overrides['imgsz'] = self.input_size
            self.detector.overrides['half'] = False
            self.detector.overrides['device'] = 'cpu'
            
            logger.info("✅ Production YOLO initialized: %s (size: %d)", model_path.name, self.input_size)
        except Exception as e:
            logger.error("❌ YOLO initialization failed: %s", e)
    
    def detect_persons(self, frame: np.ndarray, detection_freq: int = 4, 
                      conf_threshold: float = 0.4) -> List[List[int]]:
        """Detección con cache optimizado"""
        if self.detector is None:
            return []
        
        self.frame_count += 1
        
        # Use cached detections for performance
        if self.frame_count % detection_freq != 0:
            return self.last_detections
        
        try:
            # Optimization: resize frame if too large for faster detection
            detection_frame = frame
            scale_factor = 1.0
            
            if frame.shape[0] > 640:
                scale_factor = 640 / frame.shape[0]
                new_height = int(frame.shape[0] * scale_factor)
                new_width = int(frame.shape[1] * scale_factor)
                detection_frame = cv2.resize(frame, (new_width, new_height))
            
            results = self.detector(detection_frame, verbose=False)
            
            persons = []
            for result in results:
                for box in result.boxes:
                    if box.cls == 0 and box.conf >= conf_threshold:  # Person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Scale back to original frame size
                        if scale_factor != 1.0:
                            x1, y1, x2, y2 = [coord / scale_factor for coord in [x1, y1, x2, y2]]
                        
                        persons.append([int(x1), int(y1), int(x2), int(y2)])
            
            self.last_detections = persons
            return persons
            
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
    """Procesador V4 usando imports directos para máximo rendimiento"""
    
    def __init__(self, model_path: str, preset: str = 'ultra_fast', backend: str = 'pytorch'):
        self.model_path = model_path
        self.preset = preset
        self.backend = backend
        self.config = PRODUCTION_PRESETS[preset]
        self.hardware = detect_optimized_hardware()
        
        # Components
        self.yolo_detector = ProductionYOLODetector(input_size=self.config['yolo_size'])
        self.inference_engine = ProductionInferenceEngine(model_path, backend)
        
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
        logger.info("   Threading: %s (%d threads)", 
                   "enabled" if self.config['enable_threading'] else "disabled",
                   self.config['thread_count'])
    
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
            conf_threshold=0.4
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
        """Process single person using EXACT ConvNeXt logic"""
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
            
            # Return 2D coordinates
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

def draw_pose(image: np.ndarray, pose_2d: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)):
    """Draw pose using standard COCO skeleton"""
    # COCO 18-point skeleton connections
    skeleton = [
        (10, 9), (9, 8), (8, 11), (8, 14),
        (11, 12), (12, 13), (14, 15), (15, 16),
        (11, 4), (14, 1), (0, 4), (0, 1),
        (4, 5), (5, 6), (1, 2), (2, 3)
    ]
    
    # Draw joints
    for joint in pose_2d:
        x, y = int(joint[0]), int(joint[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 3, color, -1)
    
    # Draw skeleton
    for start_idx, end_idx in skeleton:
        if start_idx < len(pose_2d) and end_idx < len(pose_2d):
            start_point = (int(pose_2d[start_idx][0]), int(pose_2d[start_idx][1]))
            end_point = (int(pose_2d[end_idx][0]), int(pose_2d[end_idx][1]))
            
            # Check if both points are within image bounds
            if (0 <= start_point[0] < image.shape[1] and 0 <= start_point[1] < image.shape[0] and
                0 <= end_point[0] < image.shape[1] and 0 <= end_point[1] < image.shape[0]):
                cv2.line(image, start_point, end_point, color, 2)

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt Pose Real-time Production Pipeline (Optimized)')
    parser.add_argument('--input', type=str, default='0', 
                       help='Input source: webcam index (0), video file path, or image path')
    parser.add_argument('--model', type=str, default='models/model_opt_S.pth',
                       help='Path to the ConvNeXt pose model')
    parser.add_argument('--preset', choices=list(PRODUCTION_PRESETS.keys()), 
                       default='ultra_fast', help='Performance preset')
    parser.add_argument('--backend', choices=['pytorch', 'onnx', 'tflite'], 
                       default='pytorch', help='Inference backend')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Path to save output video')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable video display (useful for benchmarking)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ProductionV4Processor(args.model, args.preset, args.backend)
    
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
