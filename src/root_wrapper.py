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
import logging

logger = logging.getLogger(__name__)
def apply_torchvision_patch():
    """Aplica un parche a torchvision para compatibilidad con código antiguo"""
    import torchvision
    import torchvision.models.resnet as resnet_module
    
    # Solo aplicar si no existe model_urls
    if not hasattr(resnet_module, 'model_urls'):
        # Recrear model_urls como existía en versiones antiguas
        resnet_module.model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        }
        print("[INFO] Se aplicó parche de compatibilidad para torchvision")

class RootNetWrapper:
    def __init__(self, rootnet_path, checkpoint_path):
        self.rootnet_path = rootnet_path
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.cfg = None
        self._original_path = None
        
    @contextmanager
    def _isolated_import(self):
        """Context manager para aislar imports de RootNet."""
        # Guardar estado actual
        self._original_path = sys.path.copy()
        original_modules = set(sys.modules.keys())
        
        try:
            apply_torchvision_patch()
            # Añadir paths de RootNet
            if self.rootnet_path not in sys.path:
                sys.path.insert(0, self.rootnet_path)
                sys.path.insert(0, os.path.join(self.rootnet_path, 'main'))
                sys.path.insert(0, os.path.join(self.rootnet_path, 'data'))
                sys.path.insert(0, os.path.join(self.rootnet_path, 'common'))
            
            yield
            
        finally:
            # Restaurar estado
            sys.path = self._original_path
            # Remover módulos de RootNet para evitar conflictos
            new_modules = set(sys.modules.keys()) - original_modules
            for module in new_modules:
                if any(path in module for path in ['rootnet', 'main', 'data', 'common']):
                    sys.modules.pop(module, None)
    
    def load_model(self, use_gpu=True):
        """Carga RootNet en contexto aislado."""
        with self._isolated_import():
            try:
                # Importar módulos necesarios
                spec = importlib.util.spec_from_file_location(
                    "rootnet_model", 
                    os.path.join(self.rootnet_path, 'main', "model.py")
                )
                rootnet_model_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_model_module)
                
                spec = importlib.util.spec_from_file_location(
                    "rootnet_config", 
                    os.path.join(self.rootnet_path, 'main', "config.py")
                )
                rootnet_config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_config_module)
                
                # Crear modelo siguiendo el mismo patrón que demo.py
                self.cfg = rootnet_config_module.cfg
                model = rootnet_model_module.get_pose_net(self.cfg, is_train=False)
                
                if use_gpu and torch.cuda.is_available():
                    from torch.nn.parallel.data_parallel import DataParallel
                    model = DataParallel(model).cuda()
                
                # Cargar checkpoint - el archivo es un checkpoint de PyTorch, no un TAR real
                checkpoint = torch.load(self.checkpoint_path, 
                                       map_location='cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                # Obtener el state_dict del checkpoint
                state_dict = checkpoint.get('network', checkpoint)
                #Detectar y eliminar 'module.' si existe en las claves
                if not(use_gpu and torch.cuda.is_available()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        new_key = k.replace('module.', '')  # Eliminar 'module.' si existe
                        new_state_dict[new_key] = v
                    state_dict = new_state_dict

                    if hasattr(model, 'module'):
                        model.module.cfg = self.cfg
                    else:
                        model.cfg = self.cfg

                    original_forward = model.forward if not hasattr(model, 'module') else model.module.forward

                    def cpu_compatible_forward(self, input_img, k_value, target=None):
                        fm = self.backbone(input_img)

                        root_net = self.root
                        device = fm.device
                        
                        xy = root_net.deconv_layers(fm)
                        xy = root_net.xy_layer(xy)
                        xy = xy.view(-1,1, self.cfg.output_shape[0] * self.cfg.output_shape[1])
                        xy = F.softmax(xy, 2)
                        xy = xy.view(-1,1,self.cfg.output_shape[0],self.cfg.output_shape[1])

                        hm_x = xy.sum(dim=(2))
                        hm_y = xy.sum(dim=(3))

                        coord_x = hm_x * torch.arange(self.cfg.output_shape[1], device=device).float()
                        coord_y = hm_y * torch.arange(self.cfg.output_shape[0], device=device).float()

                        coord_x = coord_x.sum(dim= 2)
                        coord_y = coord_y.sum(dim= 2)

                        img_feat = torch.mean(fm.view(fm.size(0), fm.size(1), fm.size(2)*fm.size(3)), dim=2)
                        img_feat = torch.unsqueeze(img_feat, 2); img_feat = torch.unsqueeze(img_feat, 3);
                        gamma = root_net.depth_layer(img_feat)
                        gamma = gamma.view(-1,1)
                        depth = gamma * k_value.view(-1,1)

                        coord = torch.cat((coord_x, coord_y, depth), dim=1)
                        return coord
                    if hasattr(model, 'module'):
                        model.module.forward = types.MethodType(cpu_compatible_forward, model.module)
                    else:
                        model.forward = types.MethodType(cpu_compatible_forward, model)
                    logger.info("🚀 Applied CPU compatibility patch to RootNet")


                model.load_state_dict(state_dict)
                
                self.model = model.eval()
                logger.info("✅ RootNet loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load RootNet: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
    
    def predict_depth(self, img_patch, bbox, focal=[1500, 1500]):
        """Predice profundidad usando RootNet con la misma lógica que demo.py"""
        if self.model is None or self.cfg is None:
            return self._fallback_depth(bbox)
        
        try:
            with self._isolated_import():
                # Importar módulos para procesamiento
                spec = importlib.util.spec_from_file_location(
                    "rootnet_utils", 
                    os.path.join(self.rootnet_path, 'common', "utils", "pose_utils.py")
                )
                rootnet_utils = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_utils)
                
                spec = importlib.util.spec_from_file_location(
                    "rootnet_dataset", 
                    os.path.join(self.rootnet_path, 'data', "dataset.py")
                )
                rootnet_dataset = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rootnet_dataset)
                
                # Preparar imagen siguiendo el mismo proceso que demo.py
                transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)
                ])
                
                # Procesar bbox como en demo.py
                processed_bbox = rootnet_utils.process_bbox(np.array(bbox), 
                                                           img_patch.shape[1], img_patch.shape[0])
                if processed_bbox is None:
                    logger.warning(f"processed_bbox is None for bbox: {bbox}. Using fallback depth.")
                    return self._fallback_depth(bbox)
                img, img2bb_trans = rootnet_dataset.generate_patch_image(img_patch, 
                                                                        processed_bbox, False, 0.0)
                img = transform(img)
                
                # Calcular k_value como en demo.py
                k_value = np.array([
                    math.sqrt(self.cfg.bbox_real[0] * self.cfg.bbox_real[1] * 
                             focal[0] * focal[1] / (processed_bbox[2] * processed_bbox[3]))
                ]).astype(np.float32)
                
                # Preparar tensores
                if torch.cuda.is_available():
                    img = img.cuda()[None,:,:,:]
                    k_value = torch.FloatTensor([k_value]).cuda()[None,:]
                else:
                    img = img[None,:,:,:]
                    k_value = torch.FloatTensor([k_value])[None,:]
                
                # Ejecutar modelo
                with torch.no_grad():
                    root_3d = self.model(img, k_value)  # [x,y,z] donde z es la profundidad en mm
                root_depth = root_3d[0, 2].cpu().numpy()
                
                return root_depth
                
        except Exception as e:
            logger.warning(f"Error in RootNet prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_depth(bbox)
    
    def _fallback_depth(self, bbox):
        """Fallback basado en área del bbox."""
        bbox_area = bbox[2] * bbox[3]
        estimated_depth = 3000.0 / np.sqrt(bbox_area + 1e-6)
        return max(min(estimated_depth, 5000.0), 500.0)