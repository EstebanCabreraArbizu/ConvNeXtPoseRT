# ConvNeXt Pose Real-time with Quantized YOLO Integration

High-performance real-time pose estimation pipeline combining ConvNeXt pose networks with quantized YOLO detection for optimal latency and accuracy.

## 🚀 Key Features

### ⚡ Performance Optimizations
- **Quantized YOLO Integration**: ONNX INT8 quantized YOLO models for 2-3x faster person detection
- **Multi-Backend Support**: PyTorch, ONNX, and TFLite backends for pose estimation
- **Intelligent Caching**: Smart detection caching with adaptive frame skipping
- **Hardware-Aware Configuration**: Automatic optimization based on available hardware

### 🎯 Detection & Pose Estimation
- **Fast Person Detection**: Quantized YOLO11 models with sub-10ms detection times
- **2D/3D Pose Estimation**: Optional RootNet integration for 3D depth estimation
- **Multi-Person Support**: Efficient handling of multiple persons per frame
- **Production-Ready**: Robust error handling and fallback mechanisms

### 🔧 Technical Features
- **Hybrid YOLO Processing**: ONNX quantized with PyTorch fallback
- **Adaptive Preprocessing**: Hardware-optimized image processing
- **Thread Pool Management**: Controlled parallel processing for pose estimation
- **Memory Efficient**: Optimized memory usage and garbage collection

## 📊 Performance Results

| Backend | Detection (ms) | Pose (ms) | Total FPS | Notes |
|---------|---------------|-----------|-----------|-------|
| ONNX Quantized + PyTorch | 3-8 | 15-25 | 25-30 | Best balance |
| PyTorch + PyTorch | 8-15 | 15-25 | 18-25 | Full PyTorch |
| ONNX + ONNX | 3-8 | 8-15 | 35-45 | Fastest |

*Results on typical consumer hardware (CPU-optimized)*

## 🛠️ Installation

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install ultralytics
pip install onnxruntime
pip install opencv-python
pip install numpy

# Optional: GPU acceleration
pip install onnxruntime-gpu

# Optional: TensorFlow Lite support
pip install tensorflow
```

### Project Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd ConvNeXtPoseRT

# Download and setup YOLO models
python setup_models.py

# Test the integration
python test_integration.py
```

## 🚀 Quick Start

### Basic Usage
```bash
# Webcam with ultra-fast preset (quantized YOLO)
python main.py --preset ultra_fast --backend pytorch

# Video file with quality focus
python main.py --input video.mp4 --preset quality_focused --backend onnx

# Enable 3D pose estimation (requires RootNet)
python main.py --preset speed_balanced --backend pytorch --enable_3d
```

### Performance Presets

#### Ultra Fast (15+ FPS)
```bash
python main.py --preset ultra_fast --backend pytorch
```
- Target: 15+ FPS
- YOLO: 320px quantized
- Detection: Every 4th frame
- Best for: Real-time applications

#### Speed Balanced (12+ FPS)
```bash
python main.py --preset speed_balanced --backend onnx
```
- Target: 12+ FPS
- YOLO: 416px quantized
- Detection: Every 3rd frame
- Best for: Quality-speed balance

#### Quality Focused (10+ FPS)
```bash
python main.py --preset quality_focused --backend onnx --enable_3d
```
- Target: 10+ FPS
- YOLO: 512px quantized
- Detection: Every 2nd frame
- Best for: High-quality results

## 🔬 Advanced Usage

### Model Setup and Conversion
```bash
# Download and convert YOLO models
python setup_models.py

# Convert specific model
python setup_models.py --model yolo11s.pt

# Skip quantization (if not supported)
python setup_models.py --no-quantize

# List available models
python setup_models.py --list
```

### Performance Benchmarking
```bash
# Run performance benchmarks
python demo_quantized.py --benchmark --iterations 100

# Real-time demo with metrics
python demo_quantized.py --demo --duration 30

# Quick integration test
python demo_quantized.py
```

### Custom Configuration
```python
from main import ProductionV4Processor

# Custom processor
processor = ProductionV4Processor(
    model_path='models/model_opt_S.pth',
    preset='ultra_fast',
    backend='pytorch',
    enable_3d=False
)

# Process frame
poses, stats = processor.process_frame(frame)
print(f"FPS: {stats['instant_fps']:.1f}")
```

## 🧪 Testing

### Integration Tests
```bash
# Run all integration tests
python test_integration.py

# Test individual components
python -c "from main import ProductionYOLODetector; print('✅ YOLO OK')"
python -c "from main import ProductionInferenceEngine; print('✅ Pose OK')"
```

### Performance Validation
```bash
# Benchmark YOLO detection only
python demo_quantized.py --benchmark

# Full pipeline benchmark
python demo_quantized.py --benchmark --iterations 50
```

## 📁 Project Structure

```
ConvNeXtPoseRT/
├── main.py                 # Main pipeline with quantized YOLO
├── setup_models.py         # Model download and conversion
├── demo_quantized.py       # Performance demo and benchmarks
├── test_integration.py     # Integration tests
├── src/
│   ├── convnext_wrapper.py # ConvNeXt model wrapper
│   └── root_wrapper.py     # RootNet 3D wrapper
├── models/                 # Model files (gitignored)
│   ├── model_opt_S.pth     # ConvNeXt pose model
│   ├── yolo11n.pt          # YOLO PyTorch model
│   └── *.onnx              # Quantized ONNX models
├── configs/                # Configuration files
├── tests/                  # Test data and scripts
└── docs/                   # Documentation
```

## ⚙️ Configuration

### Hardware Optimization
The system automatically detects and optimizes for your hardware:

- **GPU Available**: Uses CUDA providers, increased batch sizes
- **CPU Only**: Optimizes thread counts, uses quantized models
- **Memory Constrained**: Reduces cache sizes, enables aggressive optimization

### Manual Tuning
```python
# Custom YOLO detector
detector = ProductionYOLODetector(
    model_path='models/yolo11n.pt',
    input_size=640  # Adjust based on speed/accuracy needs
)

# Custom inference engine
engine = ProductionInferenceEngine(
    model_path='models/model_opt_S.pth',
    backend='onnx'  # pytorch, onnx, tflite
)
```

## 🐛 Troubleshooting

### Common Issues

#### ONNX Model Not Found
```bash
# Download and convert models
python setup_models.py
```

#### Slow Performance
```bash
# Use faster preset
python main.py --preset ultra_fast

# Check hardware optimization
python demo_quantized.py  # Shows detected configuration
```

#### 3D Estimation Fails
```bash
# Check RootNet availability
python -c "from main import ROOTNET_AVAILABLE; print(f'RootNet: {ROOTNET_AVAILABLE}')"

# Use 2D mode
python main.py --preset ultra_fast  # 3D disabled by default
```

### Performance Issues
1. **Low FPS**: Try `ultra_fast` preset or reduce input resolution
2. **High Memory**: Disable threading or reduce cache sizes
3. **Quantization Errors**: Use `--no-quantize` flag in setup

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_integration.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ConvNeXt Pose**: Original pose estimation network
- **YOLO**: Real-time object detection
- **ONNX Runtime**: Optimized inference engine
- **RootNet**: 3D pose estimation depth network

## 📈 Changelog

### v2.0.0 - Quantized YOLO Integration
- ✅ Added ONNX quantized YOLO support
- ✅ Improved detection latency by 2-3x
- ✅ Enhanced hardware-aware optimization
- ✅ Added comprehensive benchmarking tools
- ✅ Robust fallback mechanisms

### v1.0.0 - Initial Release
- ✅ ConvNeXt pose estimation pipeline
- ✅ Multi-backend support
- ✅ RootNet 3D integration
- ✅ Production-ready optimization