import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns

class DentalRadiographCNN(nn.Module):
    """
    Custom CNN specifically designed for dental caries detection in radiographs
    Optimized for 500-600 image datasets with ~75K parameters
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(DentalRadiographCNN, self).__init__()
        
        # Feature extraction backbone
        # Designed to capture dental structures and caries patterns
        
        # Block 1: Initial feature extraction (224x224 -> 112x112)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Block 2: Tooth structure features (112x112 -> 56x56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.6)
        )
        
        # Block 3: Detailed dental features (56x56 -> 28x28)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.7)
        )
        
        # Block 4: High-level pattern recognition (28x28 -> 14x14)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.8)
        )
        
        # Global Average Pooling - reduces overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Attention mechanism for caries localization
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Feature extraction
        x1 = self.block1(x)      # [B, 32, 112, 112]
        x2 = self.block2(x1)     # [B, 64, 56, 56]
        x3 = self.block3(x2)     # [B, 128, 28, 28]
        x4 = self.block4(x3)     # [B, 256, 14, 14]
        
        # Attention mechanism
        attention_weights = self.attention(x4)  # [B, 1, 14, 14]
        attended_features = x4 * attention_weights  # Element-wise multiplication
        
        # Global pooling and classification
        pooled_features = self.global_avg_pool(attended_features)  # [B, 256, 1, 1]
        flattened = pooled_features.view(pooled_features.size(0), -1)  # [B, 256]
        
        # Classification
        output = self.classifier(flattened)
        
        return {
            'classification': output,
            'attention': F.interpolate(attention_weights, size=(224, 224), mode='bilinear', align_corners=False)
        }

class DentalRadiographDataset(Dataset):
    """Optimized dataset for dental radiographs with smart augmentation"""
    
    def __init__(self, image_paths, labels, transform=None, augment_caries=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment_caries = augment_caries
        
        # Balance dataset by augmenting caries cases
        if augment_caries:
            self.balanced_paths, self.balanced_labels = self._balance_dataset()
        else:
            self.balanced_paths = image_paths
            self.balanced_labels = labels
    
    def _balance_dataset(self):
        """Balance dataset by creating more augmented versions of caries images"""
        caries_paths = [p for p, l in zip(self.image_paths, self.labels) if l == 1]
        no_caries_paths = [p for p, l in zip(self.image_paths, self.labels) if l == 0]
        
        # Calculate augmentation factor
        if len(caries_paths) < len(no_caries_paths):
            augment_factor = len(no_caries_paths) // len(caries_paths)
            balanced_paths = no_caries_paths + (caries_paths * augment_factor)
            balanced_labels = [0] * len(no_caries_paths) + [1] * (len(caries_paths) * augment_factor)
        else:
            balanced_paths = self.image_paths
            balanced_labels = self.labels
        
        return balanced_paths, balanced_labels
    
    def __len__(self):
        return len(self.balanced_paths)
    
    def __getitem__(self, idx):
        image_path = self.balanced_paths[idx]
        label = self.balanced_labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image, 
            'label': torch.tensor(label, dtype=torch.long),
            'path': image_path
        }

def get_dental_transforms():
    """Transforms specifically optimized for dental radiographs"""
    
    train_transform = transforms.Compose([
        # Resize to standard input size
        transforms.Resize((224, 224)),
        
        # Dental-specific augmentations
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3,    # Radiograph exposure variations
                contrast=0.4,      # Different imaging settings
                saturation=0.1,    # Minimal saturation change
                hue=0.05          # Slight hue variations
            )
        ], p=0.8),
        
        # Geometric augmentations (dental positioning variations)
        transforms.RandomHorizontalFlip(p=0.5),  # Left/right teeth
        transforms.RandomRotation(degrees=12, fill=0),  # Patient positioning
        transforms.RandomAffine(
            degrees=8,
            translate=(0.1, 0.1),   # Slight positioning changes
            scale=(0.9, 1.1),       # Zoom variations
            shear=3,                # Minor perspective changes
            fill=0
        ),
        
        # Imaging quality variations
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
        ], p=0.3),
        
        # Random crop for focus on different tooth regions
        transforms.RandomResizedCrop(
            224, 
            scale=(0.85, 1.0),      # Keep most of the tooth visible
            ratio=(0.8, 1.2)        # Dental proportions
        ),
        
        transforms.ToTensor(),
        
        # Normalization optimized for grayscale-like radiographs
        transforms.Normalize(
            mean=[0.456, 0.456, 0.456],  # Adjusted for radiograph characteristics
            std=[0.224, 0.224, 0.224]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.456, 0.456, 0.456],
            std=[0.224, 0.224, 0.224]
        )
    ])
    
    return train_transform, val_transform

def train_dental_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Training function optimized for dental caries detection"""
    
    # Calculate class weights for imbalanced dataset
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['label'].tolist())
    
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = (1.0 / class_counts.float()).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with appropriate settings for small dataset
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001,           # Start with higher LR, will reduce
        weight_decay=1e-3,  # Strong regularization
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training tracking
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 15
    
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs['classification'], labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs['classification'], 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_true_labels = []
        val_probabilities = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs['classification'], labels)
                val_loss += loss.item()
                
                probs = F.softmax(outputs['classification'], dim=1)
                _, predicted = torch.max(outputs['classification'], 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
                val_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of caries
        
        # Calculate metrics
        val_acc = 100 * sum(p == t for p, t in zip(val_predictions, val_true_labels)) / len(val_true_labels)
        
        # Calculate F1 score
        tp = sum(p == 1 and t == 1 for p, t in zip(val_predictions, val_true_labels))
        fp = sum(p == 1 and t == 0 for p, t in zip(val_predictions, val_true_labels))
        fn = sum(p == 0 and t == 1 for p, t in zip(val_predictions, val_true_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(f1_score * 100)
        
        # Learning rate update
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {f1_score*100:.2f}%')
        print(f'LR: {current_lr:.6f}')
        
        # Save best model based on F1 score (better for medical applications)
        if f1_score > best_val_f1:
            best_val_f1 = f1_score
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': f1_score,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, 'best_dental_caries_model.pth')
            print(f'🦷 New best model saved! Val Acc: {val_acc:.2f}%, Val F1: {f1_score*100:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
        
        print('-' * 70)
    
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%, Best F1: {best_val_f1*100:.2f}%')
    return train_losses, val_accuracies, val_f1_scores

def load_dental_dataset(data_dir):
    """Load dental radiograph dataset from organized directories"""
    
    image_paths = []
    labels = []
    
    # Supported image formats
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    print("Loading images...")
    
    # Load caries images
    caries_dir = os.path.join(data_dir, 'caries')
    if os.path.exists(caries_dir):
        caries_count = 0
        for ext in extensions:
            files = glob.glob(os.path.join(caries_dir, ext))
            files.extend(glob.glob(os.path.join(caries_dir, ext.upper())))
            image_paths.extend(files)
            labels.extend([1] * len(files))
            caries_count += len(files)
        print(f"Found {caries_count} caries images")
    
    # Load no caries images
    no_caries_dir = os.path.join(data_dir, 'no_caries')
    if os.path.exists(no_caries_dir):
        no_caries_count = 0
        for ext in extensions:
            files = glob.glob(os.path.join(no_caries_dir, ext))
            files.extend(glob.glob(os.path.join(no_caries_dir, ext.upper())))
            image_paths.extend(files)
            labels.extend([0] * len(files))
            no_caries_count += len(files)
        print(f"Found {no_caries_count} no caries images")
    
    print(f"Total images loaded: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("❌ No images found! Please check your directory structure:")
        print(f"{data_dir}/")
        print("├── caries/     (images with dental caries)")
        print("└── no_caries/  (images without caries)")
        return None, None
    
    return image_paths, labels

def visualize_predictions_with_attention(model, image_path, device='cuda'):
    """Visualize model predictions with attention maps for dental radiographs"""
    
    _, val_transform = get_dental_transforms()
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image)
    
    input_tensor = val_transform(original_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Get prediction
        probabilities = F.softmax(outputs['classification'], dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        caries_prob = probabilities[0, 1].item()
        
        # Get attention map
        attention_map = outputs['attention'][0, 0].cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original radiograph
    axes[0, 0].imshow(original_array, cmap='gray')
    axes[0, 0].set_title('Original Dental Radiograph', fontsize=12)
    axes[0, 0].axis('off')
    
    # Attention heatmap
    axes[0, 1].imshow(attention_map, cmap='hot', interpolation='bilinear')
    axes[0, 1].set_title('Model Attention Map\n(Potential Caries Regions)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Overlay
    axes[1, 0].imshow(original_array, cmap='gray', alpha=0.7)
    axes[1, 0].imshow(attention_map, cmap='hot', alpha=0.4, interpolation='bilinear')
    axes[1, 0].set_title('Attention Overlay', fontsize=12)
    axes[1, 0].axis('off')
    
    # Prediction summary
    axes[1, 1].axis('off')
    prediction_text = f"""
    DIAGNOSIS PREDICTION
    
    Predicted Class: {'🦷 CARIES DETECTED' if predicted_class == 1 else '✅ NO CARIES'}
    
    Confidence: {confidence:.1%}
    Caries Probability: {caries_prob:.1%}
    
    Clinical Note:
    {'⚠️  Recommend further examination' if caries_prob > 0.3 else '✅ Low caries risk'}
    """
    
    axes[1, 1].text(0.1, 0.5, prediction_text, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_class, confidence, attention_map

def main():
    """Main training pipeline for dental caries detection"""
    
    # Configuration
    DATA_DIR = "dental_data"  # Change to your directory
    BATCH_SIZE = 16           # Optimal for 500-600 images
    NUM_EPOCHS = 100
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🔧 Using device: {device}')
    
    # Load dataset
    image_paths, labels = load_dental_dataset(DATA_DIR)
    if image_paths is None:
        return
    
    # Train-validation split with stratification
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Get transforms
    train_transform, val_transform = get_dental_transforms()
    
    # Create datasets
    train_dataset = DentalRadiographDataset(train_paths, train_labels, 
                                          transform=train_transform, augment_caries=True)
    val_dataset = DentalRadiographDataset(val_paths, val_labels, 
                                        transform=val_transform, augment_caries=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    
    # Initialize model
    print("🦷 Initializing Dental Caries Detection Model...")
    model = DentalRadiographCNN(num_classes=2, dropout_rate=0.4)
    model.to(device)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Parameters per image ratio: {total_params / len(image_paths):.1f}")
    
    # Train model
    print("🚀 Starting training...")
    train_losses, val_accuracies, val_f1_scores = train_dental_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    # Plot training history
    plot_training_metrics(train_losses, val_accuracies, val_f1_scores)
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_dental_caries_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("🎯 Final Model Performance:")
    print(f"Best Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Best Validation F1 Score: {checkpoint['val_f1']*100:.2f}%")

def plot_training_metrics(train_losses, val_accuracies, val_f1_scores):
    """Plot comprehensive training metrics"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Training loss
    axes[0].plot(train_losses, 'b-', linewidth=2)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Validation accuracy
    axes[1].plot(val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
    axes[1].set_title('Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # F1 Score
    axes[2].plot(val_f1_scores, 'r-', linewidth=2, label='Validation F1')
    axes[2].set_title('Validation F1 Score', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score (%)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()