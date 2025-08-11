import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

# Import your existing model
from model import VGG16CariesDetectionNet, CariesDataset, get_transforms, train_model

def load_images_from_directory(data_dir):
    """
    Load images from directory structure:
    data_dir/
    ├── caries/        (images with caries)
    └── no_caries/     (images without caries)
    
    Returns:
        image_paths: List of image file paths
        labels: List of corresponding labels (1 for caries, 0 for no caries)
    """
    image_paths = []
    labels = []
    
    # Supported image extensions
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    
    # Load caries images (label = 1)
    caries_dir = os.path.join(data_dir, 'caries')
    if os.path.exists(caries_dir):
        for ext in extensions:
            files = glob.glob(os.path.join(caries_dir, ext))
            image_paths.extend(files)
            labels.extend([1] * len(files))
        print(f"Found {len([l for l in labels if l == 1])} caries images")
    else:
        print(f"Warning: {caries_dir} directory not found")
    
    # Load no caries images (label = 0)
    no_caries_dir = os.path.join(data_dir, 'no_caries')
    if os.path.exists(no_caries_dir):
        start_idx = len(image_paths)
        for ext in extensions:
            files = glob.glob(os.path.join(no_caries_dir, ext))
            image_paths.extend(files)
            labels.extend([0] * len(files))
        print(f"Found {len(labels) - start_idx} no caries images")
    else:
        print(f"Warning: {no_caries_dir} directory not found")
    
    print(f"Total images loaded: {len(image_paths)}")
    return image_paths, labels

def main():
    """Main training pipeline"""
    
    # Configuration
    DATA_DIR = "dental_data"  # Change this to your data directory path
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load images
    print("Loading images...")
    image_paths, labels = load_images_from_directory(DATA_DIR)
    
    if len(image_paths) == 0:
        print("No images found! Please check your directory structure:")
        print(f"Expected structure:")
        print(f"{DATA_DIR}/")
        print(f"├── caries/        (images with caries)")
        print(f"└── no_caries/     (images without caries)")
        return
    
    # Split data into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = CariesDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CariesDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing VGG-16 model...")
    model = VGG16CariesDetectionNet(
        num_classes=2, 
        pretrained=True, 
        enable_localization=True,
        input_size=224
    )
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    
    print("Training completed!")
    print(f"Best model saved as: best_vgg16_caries_model.pth")
    print(f"Final validation accuracy: {max(val_accuracies):.2f}%")
    
    # Save final model as well
    torch.save(model.state_dict(), 'final_vgg16_caries_model.pth')
    print("Final model saved as: final_vgg16_caries_model.pth")

if __name__ == "__main__":
    main()