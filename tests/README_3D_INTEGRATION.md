# ConvNeXt Pose Real-time with 3D Support 🎯

Pipeline de estimación de poses en tiempo real optimizado con soporte completo para **poses 3D** usando RootNet.

## 🚀 Características Principales

### ✅ Completado
- **Pipeline optimizado**: Imports directos del repo ConvNeXt (sin wrappers)
- **Rendimiento excepcional**: 12+ FPS (PyTorch), 29+ FPS (ONNX)
- **Multi-backend**: PyTorch, ONNX, TFLite
- **Poses 2D precisas**: Esqueleto completo de 18 puntos
- **🆕 Soporte 3D**: Integración con RootNet para estimación de profundidad
- **Threading inteligente**: Procesamiento paralelo controlado
- **Frame skipping**: Optimización dinámica de rendimiento

### 🔧 Arquitectura Optimizada
- **Sin wrappers**: Imports directos para máximo rendimiento
- **Cache inteligente**: Detección y poses optimizadas
- **Gestión de memoria**: Buffer reutilizable para frames
- **Hardware adaptativo**: Configuración automática de threads

## 📊 Rendimiento Comparativo

| Modo | Backend | FPS Promedio | Uso |
|------|---------|--------------|-----|
| 2D | PyTorch | ~12.9 FPS | Máxima velocidad |
| 2D | ONNX | ~29 FPS | Ultra rápido |
| 3D | PyTorch + RootNet | ~10 FPS | Poses completas 3D |
| 3D | ONNX + RootNet | ~20 FPS | 3D optimizado |

## 🎮 Uso Rápido

### Modo 2D (Ultrarrápido)
```bash
# Cámara web en tiempo real
python main.py --preset ultra_fast --backend onnx

# Video con calidad
python main.py --input video.mp4 --preset quality_focused --backend pytorch
```

### 🆕 Modo 3D (Con RootNet)
```bash
# Poses 3D completas en tiempo real
python main.py --preset speed_balanced --backend pytorch --enable_3d

# Poses 3D con ONNX (más rápido)
python main.py --preset ultra_fast --backend onnx --enable_3d --input 0
```

## 📋 Presets Disponibles

### `ultra_fast` 
- **FPS objetivo**: 15+ FPS
- **Uso**: Velocidad máxima, detección básica
- **3D**: Disponible (10+ FPS)

### `speed_balanced`
- **FPS objetivo**: 12+ FPS  
- **Uso**: Balance perfecto velocidad/calidad
- **3D**: Recomendado (8+ FPS)

### `quality_focused`
- **FPS objetivo**: 10+ FPS
- **Uso**: Máxima precisión, sin threading
- **3D**: Alta calidad (6+ FPS)

## 🛠️ Instalación y Configuración

### 1. Requisitos del Sistema
```bash
# Dependencias básicas
pip install torch torchvision opencv-python numpy
pip install ultralytics  # Para YOLO
pip install onnxruntime  # Para backend ONNX (opcional)
```

### 2. Estructura de Proyectos
```
Repository-Projects/
├── ConvNeXtPoseRT/          # Este proyecto
│   ├── main.py              # Pipeline principal con 3D
│   ├── models/              # Modelos ConvNeXt y YOLO
│   └── src/
│       └── root_wrapper.py  # Wrapper de RootNet
├── ConvNeXtPose/            # Proyecto ConvNeXt original
└── RootNet/                 # Proyecto RootNet (para 3D)
```

### 3. Modelos Requeridos

#### ConvNeXt (Obligatorio)
- `models/model_opt_S.pth` - Modelo principal PyTorch
- `models/model_opt_S_optimized.onnx` - Modelo ONNX (opcional)

#### YOLO (Obligatorio)
- `models/yolo11n.pt` - Detección de personas

#### RootNet (Para 3D)
- Requiere proyecto RootNet en directorio hermano
- Checkpoint de RootNet (automático o `models/rootnet_model.pth`)

## 🧪 Testing

### Test Rápido
```bash
# Verificar integración 3D
python test_3d_integration.py

# Test sin inferencia (más rápido)
python test_3d_integration.py --skip_inference
```

### Benchmark de Rendimiento
```bash
# Benchmark completo 2D vs 3D
python performance_test.py --test_3d

# Comparar todos los backends
python performance_test.py --backends pytorch onnx --enable_3d
```

## 💡 Características 3D

### 🆕 Estimación de Profundidad
- **RootNet**: Estimación de profundidad de la raíz corporal
- **Coordenadas 3D**: (x, y, z) donde z es profundidad en mm
- **Visualización**: Color coding por profundidad
- **Fallback**: Automático a 2D si RootNet no está disponible

### 🎨 Visualización Mejorada
- **Esqueleto 3D**: Articulaciones con codificación de profundidad
- **Información de depth**: Muestra profundidad estimada en mm
- **Estadísticas 3D**: Estado de RootNet en tiempo real

### ⚡ Optimizaciones 3D
- **Cache inteligente**: Reutilización de estimaciones de profundidad
- **Threading controlado**: Paralelización segura de RootNet
- **Fallback robusto**: Degradación elegante a 2D

## 📖 Argumentos Completos

```bash
python main.py [opciones]

Argumentos principales:
  --input SOURCE          # Fuente: 0 (cámara), video.mp4, imagen.jpg
  --preset PRESET         # ultra_fast, speed_balanced, quality_focused
  --backend BACKEND       # pytorch, onnx, tflite
  --enable_3d             # Activar modo 3D con RootNet
  --save_video OUTPUT     # Guardar video procesado
  --no_display            # Sin ventana (para benchmarking)

Ejemplos:
  python main.py --input 0 --enable_3d
  python main.py --input video.mp4 --preset quality_focused --backend onnx --enable_3d
  python main.py --input imagen.jpg --save_video output.mp4
```

## 🔧 Optimizaciones Técnicas

### Imports Directos
- Eliminación de wrappers intermedios
- Importación directa desde ConvNeXt repo
- Máximo rendimiento sin sobrecarga

### Gestión de Memoria
- Buffer de frames reutilizable
- Cache de detecciones YOLO
- Limpieza automática de recursos

### Threading Inteligente
- Pool de threads controlado
- Timeout de seguridad (200ms)
- Fallback automático en errores

### Frame Skipping Adaptativo
- Monitoreo de rendimiento en tiempo real
- Salto inteligente de frames bajo carga
- Balance automático velocidad/calidad

## 🐛 Resolución de Problemas

### RootNet No Disponible
```
⚠️ 3D requested but RootNet not available - falling back to 2D
```
**Solución**: Verificar que el proyecto RootNet esté en directorio hermano

### Modelo No Encontrado
```
❌ ConvNeXt model not found
```
**Solución**: Copiar `model_opt_S.pth` a `models/`

### Bajo Rendimiento
- Usar backend ONNX: `--backend onnx`
- Preset más rápido: `--preset ultra_fast`
- Desactivar 3D: quitar `--enable_3d`

## 📈 Monitoreo en Tiempo Real

### Estadísticas en Pantalla
- **FPS instantáneo y promedio**
- **Tiempo de procesamiento detallado**
- **Estado del backend activo**
- **Modo 2D/3D y estado de RootNet**
- **Poses detectadas por frame**

### Controles de Teclado
- `q` o `ESC`: Salir
- `r`: Reiniciar estadísticas
- `s`: Guardar frame actual

## 🎯 Estado del Proyecto

### ✅ Completado
- [x] Pipeline optimizado sin wrappers
- [x] Soporte multi-backend (PyTorch/ONNX/TFLite)
- [x] Poses 2D precisas y estables
- [x] **Integración completa RootNet para 3D**
- [x] **Visualización 3D con depth coding**
- [x] **Pipeline unificado 2D/3D**
- [x] Threading optimizado
- [x] Tests y benchmarks

### 🎯 Objetivos Cumplidos
- **Rendimiento**: >12 FPS (2D), >8 FPS (3D)
- **Estabilidad**: Sin crashes, fallback robusto
- **Calidad**: Poses precisas con esqueleto completo
- **Usabilidad**: Interface simple, configuración automática

## 📊 Comparación con Versión Original

| Aspecto | Original | Optimizado | Mejora |
|---------|----------|------------|--------|
| FPS (2D) | ~5 FPS | ~12.9 FPS | **+158%** |
| FPS (3D) | No disponible | ~10 FPS | **🆕 Nuevo** |
| Backend | Solo PyTorch | PyTorch/ONNX/TFLite | **Multi-backend** |
| Wrappers | Sí (overhead) | No (directo) | **Eliminados** |
| 3D Support | No | Sí (RootNet) | **🆕 Completo** |
| Threading | Básico | Inteligente | **Optimizado** |

---

**🎉 Resultado Final**: Pipeline completo 2D/3D con rendimiento excepcional y arquitectura optimizada sin comprometer la estabilidad.
