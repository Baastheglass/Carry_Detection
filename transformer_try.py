from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch

# -----------------------------
# 1. Select SegFormer model
# -----------------------------
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=2,  # background + caries
    ignore_mismatched_sizes=True
)

# -----------------------------
# 2. Load COCO datasets
# -----------------------------
train_dataset = load_dataset("coco", data_dir="dataset/train", split="train")
valid_dataset = load_dataset("coco", data_dir="dataset/valid", split="train")
test_dataset  = load_dataset("coco", data_dir="dataset/test", split="train")

# Attach COCO annotation handler (train annotations shown here; repeat for val/test if needed)
coco_train = COCO("dataset/train/_annotations.coco.json")
coco_val   = COCO("dataset/valid/_annotations.coco.json")

# -----------------------------
# 3. Utility: Convert COCO anns → mask
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
# 4. Preprocessing function
# -----------------------------
def preprocess_train(example):
    image = Image.open(example["file_name"]).convert("RGB")
    mask = coco_to_mask(coco_train, example["image_id"], image.height, image.width)

    inputs = processor(images=image, segmentation_maps=mask, return_tensors="pt")
    inputs["labels"] = inputs["labels"].squeeze(0)  # remove batch dim
    return inputs

def preprocess_val(example):
    image = Image.open(example["file_name"]).convert("RGB")
    mask = coco_to_mask(coco_val, example["image_id"], image.height, image.width)

    inputs = processor(images=image, segmentation_maps=mask, return_tensors="pt")
    inputs["labels"] = inputs["labels"].squeeze(0)
    return inputs

train_dataset = train_dataset.map(preprocess_train, batched=False)
valid_dataset = valid_dataset.map(preprocess_val, batched=False)

# -----------------------------
# 5. Training setup
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"  # disable wandb/mlflow unless you want them
)

# -----------------------------
# 6. Define Trainer
# -----------------------------
def compute_metrics(eval_pred):
    # Simple pixel accuracy (you can add Dice, IoU later)
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).astype(np.float32).mean()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 7. Train!
# -----------------------------
trainer.train()
