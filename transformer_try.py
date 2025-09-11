from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datasets import load_dataset
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

#selecting model
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name,
    num_labels=2,  # caries / background
    ignore_mismatched_sizes=True  # useful if num_labels is different
)

#importing dataset splits
train_dataset = load_dataset("coco", data_dir="dataset/train", split="train")
valid_dataset = load_dataset("coco", data_dir="dataset/valid", split="train")  # still "train" inside
test_dataset  = load_dataset("coco", data_dir="dataset/test", split="train")

#preprocess for segformer
coco = COCO("dataset/train/_annotations.coco.json")

def coco_to_mask(coco, img_id, height, width):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        cat_id = ann["category_id"]
        rle = coco.annToRLE(ann)
        ann_mask = coco.annToMask(ann) * cat_id
        mask = np.maximum(mask, ann_mask)
    return mask

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

def preprocess(example):
    image = Image.open(example["file_name"]).convert("RGB")
    mask = coco_to_mask(coco, example["image_id"], image.height, image.width)

    inputs = processor(images=image, segmentation_maps=mask, return_tensors="pt")
    return inputs

train_dataset = train_dataset.map(preprocess, batched=False)
valid_dataset = valid_dataset.map(preprocess, batched=False)