# ConvNeXtPoseRT - Wrapper Actualizado

## 🚀 Resumen de Cambios

El código ha sido adaptado para usar **importaciones aisladas** del `ConvNextWrapper` con soporte para **múltiples backends** (PyTorch, ONNX, TensorFlow Lite).

## 📁 Estructura Actualizada

```
ConvNeXtPoseRT/
├── src/
│   ├── convnext_wrapper.py      # ✅ NUEVO: Wrapper unificado multi-backend
│   ├── convnext_wrapper_backup.py  # Backup del wrapper original
│   └── root_wrapper.py          # Sin cambios
├── main.py                      # ✅ ACTUALIZADO: Usa el nuevo wrapper
├── config_paths.py              # ✅ NUEVO: Configuración de paths
├── test_wrapper.py              # ✅ NUEVO: Script de pruebas
└── requirements.txt             # Sin cambios
```

## 🔧 Configuración Necesaria

### 1. Configurar Paths de ConvNextPose

Edita `config_paths.py` y configura la ruta a tu instalación de ConvNextPose:

```python
# config_paths.py
CONVNEXT_POSE_PATH = "D:/Path/To/ConvNextPose"  # ⚠️ CAMBIAR ESTO
ROOTNET_PATH = "D:/Path/To/RootNet"            # ⚠️ CAMBIAR ESTO
```

### 2. Verificar Estructura de ConvNextPose

Asegúrate de que tu ConvNextPose tenga esta estructura:

```
ConvNextPose/
├── main/
│   ├── model.py
│   └── config.py
├── data/
│   └── dataset.py
└── common/
    └── utils/
        └── pose_utils.py
```

## 🧪 Pruebas

### 1. Prueba Básica del Wrapper

```bash
python test_wrapper.py
```

Este script verifica:
- ✅ Inicialización del wrapper
- ✅ Auto-detección de ConvNextPose path
- ✅ Carga del modelo PyTorch
- ✅ Inferencia básica con datos dummy
- ✅ Carga opcional de modelos ONNX

### 2. Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv convnext_env

# Activar entorno (Windows)
convnext_env\Scripts\activate

# Activar entorno (Linux/Mac)
source convnext_env/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 🎯 Características del Nuevo Wrapper

### Multi-Backend Support

```python
# Inicializar wrapper
wrapper = ConvNextWrapper(
    model_path="models/model_opt_S.pth",
    input_size=256,
    output_size=32,
    convnext_path="/path/to/ConvNextPose"  # Opcional, auto-detecta
)

# Cargar diferentes backends
wrapper.load_pytorch_model(joint_num=18)
wrapper.load_onnx_model("exports/model.onnx")
wrapper.load_tflite_model("exports/model.tflite")

# Inferencia unificada
output = wrapper.infer(image_patch, backend='pytorch')  # o 'onnx', 'tflite'
```

### Importaciones Aisladas

- ✅ **Sin conflictos**: ConvNextPose se importa en contexto aislado
- ✅ **Limpio**: No contamina el namespace global
- ✅ **Seguro**: Restaura el PATH después de usar

### Post-procesamiento Integrado

```python
# Predicción completa con post-procesamiento
pose_2d, pose_3d = wrapper.predict_pose_full(
    original_img=frame,
    bbox=[x, y, w, h],
    backend='pytorch'
)
```

## 🔄 Compatibilidad con main.py

El wrapper mantiene compatibilidad con la interfaz existente de `main.py`:

```python
# El main.py sigue funcionando igual
engine = ProductionInferenceEngine(
    model_path="models/model_opt_S.pth",
    backend='pytorch'  # o 'onnx', 'tflite'
)
```

## ⚠️ Solución de Problemas

### Error: "ConvNextPose path not found"

1. Verifica que `config_paths.py` tenga el path correcto
2. O configura manualmente:
   ```python
   wrapper = ConvNextWrapper(
       model_path="models/model_opt_S.pth",
       convnext_path="/ruta/absoluta/a/ConvNextPose"
   )
   ```

### Error: "Failed to load ConvNextPose PyTorch"

1. Verifica que el modelo `model_opt_S.pth` exista
2. Verifica que ConvNextPose tenga la estructura correcta
3. Ejecuta `test_wrapper.py` para diagnóstico detallado

### Error: Imports no encontrados

1. Asegúrate de estar en el entorno virtual correcto
2. Instala las dependencias: `pip install -r requirements.txt`
3. Para pruebas sin GPU: configura `FORCE_CPU = True` en `config_paths.py`

## 🚀 Siguiente Paso: Integración con RootNet

El `root_wrapper.py` puede actualizarse de manera similar para mantener consistencia en el pipeline completo.

## 📝 Notas Técnicas

- **Backends soportados**: PyTorch (nativo), ONNX Runtime, TensorFlow Lite
- **Aislamiento**: Context managers para imports limpios
- **Fallbacks**: Degradación elegante si backends no están disponibles
- **Logging**: Información detallada de inicialización y errores
- **Compatibilidad**: Mantiene interfaz existente del main.py
