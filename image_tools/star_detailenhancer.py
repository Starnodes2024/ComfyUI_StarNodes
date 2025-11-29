import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

class AdaptiveDetailEnhancement:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enhancement_strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 3.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.1, 
                    "max": 1.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "face_boost": ("FLOAT", {
                    "default": 1.5, 
                    "min": 0.5, 
                    "max": 3.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "texture_boost": ("FLOAT", {
                    "default": 1.2, 
                    "min": 0.5, 
                    "max": 2.5, 
                    "step": 0.1,
                    "display": "slider"
                }),
                "noise_suppression": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_details"
    CATEGORY = '⭐StarNodes/Image And Latent'

    def detect_edges(self, image_np):
        """Erkennt Kanten im Bild"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Verschiedene Edge Detection Methoden kombinieren
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        canny = cv2.Canny(gray.astype(np.uint8), 50, 150)
        
        # Kombiniere beide Methoden
        edges = np.maximum(sobel / sobel.max(), canny / 255.0)
        return edges

    def detect_faces(self, image_np):
        """Einfache Gesichtsdetektion basierend auf Hautfarben und Strukturen"""
        # Konvertiere zu HSV für bessere Hautfarbenerkennung
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Hautfarben-Bereich (erweitert für verschiedene Hauttöne)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphologische Operationen zur Verbesserung
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Gaußscher Blur für weiche Übergänge
        skin_mask = cv2.GaussianBlur(skin_mask, (31, 31), 0)
        
        return skin_mask.astype(np.float32) / 255.0

    def detect_textures(self, image_np):
        """Erkennt Texturbereiche mit hoher lokaler Varianz"""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Lokale Standardabweichung berechnen
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        mean = cv2.filter2D(gray, -1, kernel)
        sqr_mean = cv2.filter2D(gray**2, -1, kernel)
        texture_map = np.sqrt(sqr_mean - mean**2)
        
        # Normalisieren
        texture_map = texture_map / (texture_map.max() + 1e-8)
        
        return texture_map

    def apply_unsharp_mask(self, image, mask, strength):
        """Wendet Unsharp Masking nur auf maskierte Bereiche an"""
        # Gaussian Blur
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Unsharp Mask
        unsharp = image + strength * (image - blurred)
        
        # Nur auf maskierte Bereiche anwenden
        result = image.copy()
        for c in range(3):  # RGB Kanäle
            result[:, :, c] = image[:, :, c] * (1 - mask) + unsharp[:, :, c] * mask
            
        return np.clip(result, 0, 1)

    def bilateral_denoise(self, image, strength):
        """Selektive Rauschunterdrückung mit Bilateral Filter"""
        if strength <= 0:
            return image
            
        # Konvertiere zu uint8 für OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Bilateral Filter Parameter basierend auf Stärke
        d = int(5 + strength * 5)  # Nachbarschaftsdurchmesser
        sigma_color = 50 + strength * 50  # Farbähnlichkeit
        sigma_space = 50 + strength * 50  # Räumliche Nähe
        
        denoised = cv2.bilateralFilter(image_uint8, d, sigma_color, sigma_space)
        
        return denoised.astype(np.float32) / 255.0

    def enhance_details(self, image, enhancement_strength, edge_threshold, 
                       face_boost, texture_boost, noise_suppression):
        
        # Konvertiere von torch tensor zu numpy
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
            if len(image_np.shape) == 4:
                image_np = image_np[0]  # Batch dimension entfernen
        else:
            image_np = image
        
        # Stelle sicher, dass Werte im Bereich [0,1] sind
        image_np = np.clip(image_np, 0, 1)
        
        # Konvertiere für OpenCV (braucht uint8 oder float32)
        img_for_cv = (image_np * 255).astype(np.uint8)
        img_float = image_np.astype(np.float32)
        
        # 1. Rauschunterdrückung (falls gewünscht)
        if noise_suppression > 0:
            img_float = self.bilateral_denoise(img_float, noise_suppression)
        
        # 2. Verschiedene Bereiche erkennen
        edges = self.detect_edges(img_for_cv)
        faces = self.detect_faces(img_for_cv)
        textures = self.detect_textures(img_for_cv)
        
        # 3. Adaptive Enhancement-Masken erstellen
        
        # Edge Enhancement Mask
        edge_mask = (edges > edge_threshold).astype(np.float32)
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 1.0)
        
        # Face Enhancement Mask
        face_mask = faces * face_boost
        
        # Texture Enhancement Mask  
        texture_mask = textures * texture_boost
        
        # Kombiniere alle Masken
        combined_mask = np.clip(
            edge_mask * enhancement_strength + 
            face_mask + 
            texture_mask, 
            0, 3.0
        )
        
        # 4. Selektive Schärfung anwenden
        enhanced = self.apply_unsharp_mask(img_float, combined_mask, 1.0)
        
        # 5. Zurück zu torch tensor konvertieren
        if isinstance(image, torch.Tensor):
            result = torch.from_numpy(enhanced).unsqueeze(0)  # Batch dimension hinzufügen
            if image.device.type == 'cuda':
                result = result.cuda()
        else:
            result = enhanced
            
        return (result,)

# Node Mapping für ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdaptiveDetailEnhancement": AdaptiveDetailEnhancement
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveDetailEnhancement": "⭐ Star Adaptive Detail Enhancement"
}