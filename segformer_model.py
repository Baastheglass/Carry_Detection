import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SegFormerCariesDetection:
    """SegFormer model for dental caries segmentation"""
    
    def __init__(self, model_path='./results/checkpoint-26955', processor_path='nvidia/mit-b0'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load processor from original model and fine-tuned model
        self.processor = SegformerImageProcessor.from_pretrained(processor_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, image):
        """
        Predict segmentation mask for an input image
        
        Args:
            image: PIL.Image - Input dental X-ray
            
        Returns:
            dict containing:
                - predicted_class: 0 for negative, 1 if caries detected
                - confidence: confidence score for the prediction
                - caries_prob: probability of caries
                - attention_map: segmentation heatmap highlighting caries areas
        """
        # Resize for consistent processing
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Preprocess the image
        target_size = (512, 512)
        resized_image = image.resize(target_size, Image.BILINEAR)
        
        # Process with SegFormer's processor
        inputs = self.processor(images=resized_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get segmentation mask
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Check if any pixel is classified as caries (class 1)
            has_caries = np.any(mask > 0)
            predicted_class = 1 if has_caries else 0
            
            # Calculate caries probability based on pixel ratio
            total_pixels = mask.size
            caries_pixels = np.sum(mask > 0)
            caries_ratio = caries_pixels / total_pixels if total_pixels > 0 else 0
            
            # Calculate confidence based on the certainty of the prediction
            if predicted_class == 1:
                confidence = min(0.5 + caries_ratio, 0.99)  # Scale confidence based on caries extent
            else:
                confidence = max(0.5 + (1 - caries_ratio) * 0.4, 0.7)  # Higher confidence for definite negatives
                
            # Use the mask as attention map
            attention_map = mask.astype(np.float32)
            
            # If there's any caries, normalize the map for better visualization
            if has_caries:
                attention_map = attention_map / np.max(attention_map) if np.max(attention_map) > 0 else attention_map
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'caries_prob': float(caries_ratio),
            'attention_map': attention_map
        }