import cv2
import numpy as np
import torch
import base64

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_a = target_layer.register_forward_hook(self.save_activation)
        self.hook_g = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate(self):
        if self.gradients is None or self.activations is None:
            # print("DEBUG: Grad-CAM failed - Missing gradients or activations") # Reduce log spam
            return None
            
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Get activations of the last conv layer
        activations = self.activations.detach()
        
        # Weight the channels by corresponding gradients
        # Use simple loop if batch size is 1, else simpler broadcasting
        # Assuming batch size 1 for inference
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on top
        if torch.is_tensor(heatmap):
            heatmap = heatmap.cpu().numpy()
            
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap /= np.max(heatmap) + 1e-8
        
        return heatmap
        
    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()

def generate_heatmap_overlay(heatmap, img_bgr):
    """Overlays heatmap on original image."""
    try:
        h, w = img_bgr.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        
        # Colorize
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay
        superimposed_img = heatmap * 0.4 + img_bgr * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # Encode
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return ""
