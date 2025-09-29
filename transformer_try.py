import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer
)
from pycocotools.coco import COCO

# -----------------------------
# 1. Select SegFormer model  
# -----------------------------
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name, reduce_labels=False)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=2,  # background + caries
    ignore_mismatched_sizes=True
)

# -----------------------------
# 2. Utility: Convert COCO anns → mask
# -----------------------------
def coco_to_mask(coco, img_id, height, width):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        cat_id = ann["category_id"]  # caries=1, background=0
        ann_mask = coco.annToMask(ann) * cat_id
        mask = np.maximum(mask, ann_mask)  # overlay multiple anns
    return mask

# -----------------------------
# 3. Custom COCO Dataset
# -----------------------------
class COCOSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_file, processor):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info["file_name"])

        # Load image and mask
        image = Image.open(path).convert("RGB")
        mask = coco_to_mask(self.coco, img_id, img_info["height"], img_info["width"])
        
        # Resize image and mask to consistent size
        target_size = (512, 512)
        image = image.resize(target_size, Image.BILINEAR)
        mask = Image.fromarray(mask).resize(target_size, Image.NEAREST)
        mask = np.array(mask)

        # Preprocess with SegFormer's processor
        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        
        # Remove batch dimension and ensure proper tensor types
        pixel_values = inputs["pixel_values"].squeeze(0)
        labels = inputs["labels"].squeeze(0).long()
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

# -----------------------------
# 4. Create train/val/test datasets
# -----------------------------
train_dataset = COCOSegmentationDataset(
    img_dir="dataset/train", ann_file="dataset/train/_annotations.coco.json", processor=processor
)
valid_dataset = COCOSegmentationDataset(
    img_dir="dataset/valid", ann_file="dataset/valid/_annotations.coco.json", processor=processor
)
test_dataset = COCOSegmentationDataset(
    img_dir="dataset/test", ann_file="dataset/test/_annotations.coco.json", processor=processor
)

# -----------------------------
# 5. Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,  # Reduced batch size to avoid memory issues
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    num_train_epochs=5,  # Reduced epochs for faster testing
    do_eval=True,
    save_total_limit=2,
    eval_strategy="steps",  # Fixed: changed from evaluation_strategy
    eval_steps=500,
    save_strategy="steps", 
    save_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",  # disable wandb/mlflow unless you want them
    dataloader_pin_memory=False  # Disable pin_memory for MPS compatibility
)

# -----------------------------
# 6. Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    # Flatten for proper comparison
    preds = preds.flatten()
    labels = labels.flatten()
    acc = (preds == labels).astype(np.float32).mean()
    return {"accuracy": acc}

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 8. Train!
# -----------------------------
trainer.train()
