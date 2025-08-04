import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class CariesDataset(Dataset):
    """Custom dataset for dental caries detection"""
    
    def __init__(self, image_paths, labels, masks=None, transform=None):
        """
        Args:
            image_paths: List of paths to radiograph images
            labels: List of binary labels (0: no caries, 1: caries present)
            masks: List of segmentation masks (optional, for localization)
            transform: Data augmentation transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        sample = {'image': image, 'label': torch.tensor(label, dtype=torch.long)}
        
        # Include mask if available (for segmentation training)
        if self.masks is not None:
            mask = Image.open(self.masks[idx]).convert('L')
            if self.transform:
                # Apply same transforms to mask (except normalization)
                mask_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                mask = mask_transform(mask)
            sample['mask'] = mask
        
        return sample

class VGG16CariesDetectionNet(nn.Module):
    """VGG-16 based CNN for caries detection and localization"""
    
    def __init__(self, num_classes=2, pretrained=True, enable_localization=True, input_size=224):
        super(VGG16CariesDetectionNet, self).__init__()
        
        self.enable_localization = enable_localization
        self.input_size = input_size
        
        if pretrained:
            # Load pretrained VGG16 and modify it
            vgg16 = models.vgg16(pretrained=True)
            self.features = vgg16.features
        else:
            # Build VGG-16 architecture from scratch
            self.features = self._make_vgg16_features()
        
        # Calculate the size after feature extraction
        # For 224x224 input, after 5 max pooling operations (224/32 = 7)
        feature_size = (input_size // 32) ** 2 * 512  # 7*7*512 = 25088 for 224x224
        
        # VGG-16 style classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
        # Localization head (for generating attention maps)
        if enable_localization:
            self.localization = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(128, 1, kernel_size=1),  # Single channel for attention
                nn.Sigmoid()
            )
    
    def _make_vgg16_features(self):
        """Create VGG-16 feature extraction layers"""
        layers = []
        in_channels = 3
        
        # VGG-16 configuration: number of output channels for each conv layer
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Extract features through VGG-16 backbone
        features = self.features(x)  # Shape: [batch, 512, H/32, W/32]
        
        # Classification path
        # Flatten for fully connected layers
        flattened_features = features.view(features.size(0), -1)
        class_logits = self.classifier(flattened_features)
        
        results = {'classification': class_logits}
        
        # Localization (attention map)
        if self.enable_localization:
            attention_map = self.localization(features)
            # Upsample to match input size
            attention_map = F.interpolate(attention_map, size=(self.input_size, self.input_size), 
                                        mode='bilinear', align_corners=False)
            results['attention'] = attention_map
        
        return results

class GradCAM:
    """Grad-CAM for visualizing which regions the model focuses on"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output['classification'].argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output['classification'][0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach()

def get_transforms():
    """Data augmentation and preprocessing transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Training loop for the VGG-16 caries detection model"""
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_loc = nn.MSELoss()  # For attention supervision if masks available
    
    # Use a smaller learning rate for VGG-16 as it has more parameters
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion_cls(outputs['classification'], labels)
            
            # Add localization loss if masks are available
            if 'mask' in batch and model.enable_localization:
                masks = batch['mask'].to(device)
                loc_loss = criterion_loc(outputs['attention'], masks)
                loss += 0.5 * loc_loss  # Weight the localization loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion_cls(outputs['classification'], labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs['classification'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vgg16_caries_model.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
        
        scheduler.step(avg_val_loss)
        print('-' * 50)
    
    return train_losses, val_accuracies

def visualize_predictions(model, image_path, device='cuda'):
    """Visualize model predictions with attention maps"""
    
    # Load and preprocess image
    _, val_transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Get prediction
        probabilities = F.softmax(outputs['classification'], dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # Get attention map if available
        if 'attention' in outputs:
            attention_map = outputs['attention'][0, 0].cpu().numpy()
        else:
            # Use Grad-CAM as fallback - target the last conv layer
            target_layer = None
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module  # Get the last conv layer
            
            if target_layer is not None:
                grad_cam = GradCAM(model, target_layer)
                attention_map = grad_cam.generate_cam(input_tensor, predicted_class).cpu().numpy()
            else:
                attention_map = np.zeros((224, 224))
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Radiograph')
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attention_map, cmap='hot')
    axes[1].set_title('Attention Map (Potential Caries Locations)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_image, cmap='gray', alpha=0.7)
    axes[2].imshow(attention_map, cmap='hot', alpha=0.3)
    axes[2].set_title(f'Overlay - Prediction: {"Caries" if predicted_class == 1 else "No Caries"}\nConfidence: {confidence:.2f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence, attention_map

def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model architecture summary similar to the format you provided"""
    
    def get_model_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("VGG-16 Caries Detection Model Summary:")
    print("=" * 60)
    print(f"Total trainable parameters: {get_model_parameters(model):,}")
    print("=" * 60)
    
    # Print feature extraction layers
    print("Feature extraction layers (VGG-16 backbone):")
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            print(f"conv2d_{i+1:<2} (Conv2D): {params:,} parameters")
        elif isinstance(layer, nn.MaxPool2d):
            print(f"max_pooling2d_{i+1:<2} (MaxPool2D): 0 parameters")
    
    print("\nClassification layers:")
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            print(f"dense_{i+1:<2} (Linear): {params:,} parameters")
    
    if model.enable_localization:
        print("\nLocalization layers:")
        for i, layer in enumerate(model.localization):
            if isinstance(layer, nn.Conv2d):
                params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
                print(f"loc_conv_{i+1:<2} (Conv2D): {params:,} parameters")

def main():
    """Main training and evaluation pipeline"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize VGG-16 model
    model = VGG16CariesDetectionNet(num_classes=2, pretrained=True, enable_localization=True)
    model.to(device)
    
    print("VGG-16 Model architecture:")
    print_model_summary(model)
    
    # Example usage - you'll need to replace these with your actual data paths
    try:
        train_images = [os.path.join('caries', f) for f in os.listdir('caries') 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))] + \
                   [os.path.join('without_caries', f) for f in os.listdir('without_caries') 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

        train_labels = [1] * len(os.listdir('caries')) + [0] * len(os.listdir('without_caries'))
        val_images = [os.path.join('val_caries', f) for f in os.listdir('val_caries') 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))] + \
                      [os.path.join('val_without_caries', f) for f in os.listdir('val_without_caries') 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        val_labels = [1] * len(os.listdir('val_caries')) + [0] * len(os.listdir('val_without_caries'))
        
        # Get transforms
        train_transform, val_transform = get_transforms()
        
        # Create datasets
        train_dataset = CariesDataset(train_images, train_labels, transform=train_transform)
        val_dataset = CariesDataset(val_images, val_labels, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)  # Smaller batch size for VGG-16
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        
        # Train model
        print("Starting training...")
        train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=5, device=device)
        
        # Load trained model for inference
        model.load_state_dict(torch.load('best_vgg16_caries_model.pth'))
        
        # Visualize predictions on test images
        test_image_path = 'val_caries/CamScanner 30-07-2025 23.46_1.jpg'
        if os.path.exists(test_image_path):
            predicted_class, confidence, attention_map = visualize_predictions(model, test_image_path, device)
        else:
            print(f"Test image not found: {test_image_path}")
            
    except FileNotFoundError as e:
        print(f"Data directories not found: {e}")
        print("Please ensure your data is organized in the following structure:")
        print("- caries/ (images with caries)")
        print("- without_caries/ (images without caries)")
        print("- val_caries/ (validation images with caries)")
        print("- val_without_caries/ (validation images without caries)")
        
        # Create a dummy model for demonstration
        print("\nDemonstration model created successfully!")
        print("Replace the data paths with your actual dataset to start training.")

if __name__ == "__main__":
    main()