import os
import json
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator
import torch
# import ssl
# import certifi

# ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Setup logging
setup_logger()

def clean_coco_annotations(coco_file_path, output_file_path, valid_class_ids):
    """
    Clean COCO annotations to remove problematic categories
    """
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    
    # Filter categories to only valid ones
    valid_categories = [cat for cat in coco_data['categories'] if cat['id'] in valid_class_ids]
    
    # Create mapping from old IDs to new sequential IDs
    id_mapping = {cat['id']: idx for idx, cat in enumerate(valid_categories)}
    
    # Update category IDs to be sequential starting from 0
    for i, cat in enumerate(valid_categories):
        cat['id'] = i
    
    # Filter annotations to only include valid categories
    valid_annotations = []
    for ann in coco_data['annotations']:
        if ann['category_id'] in id_mapping:
            ann['category_id'] = id_mapping[ann['category_id']]
            valid_annotations.append(ann)
    
    # Update the data
    coco_data['categories'] = valid_categories
    coco_data['annotations'] = valid_annotations
    
    # Save cleaned annotations
    with open(output_file_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    return len(valid_categories)

def register_datasets():
    """
    Register your COCO datasets with Detectron2
    """
    # Define valid class IDs (adjust based on your actual classes)
    # Remove metadata classes like collaboration tools, separators, etc.
    valid_class_ids = [5, 6, 7, 8, 9, 10, 11, 12]  # Negative, Positive, RA1-RC6
    
    # Clean and register training dataset
    train_coco_clean = "./dataset/train/_annotations_clean.coco.json"
    num_classes = clean_coco_annotations(
        "./dataset/train/_annotations.coco.json",
        train_coco_clean,
        valid_class_ids
    )
    
    register_coco_instances(
        "my_dataset_train", 
        {}, 
        train_coco_clean,
        "./dataset/train/"
    )
    
    # Clean and register validation dataset
    val_coco_clean = "./dataset/valid/_annotations_clean.coco.json"
    clean_coco_annotations(
        "./dataset/valid/_annotations.coco.json",
        val_coco_clean,
        valid_class_ids
    )
    
    register_coco_instances(
        "my_dataset_val", 
        {}, 
        val_coco_clean,
        "./dataset/valid/"
    )
    
    # Optional: register test dataset if you have one
    if os.path.exists("./dataset/test/_annotations.coco.json"):
        test_coco_clean = "./dataset/test/_annotations_clean.coco.json"
        clean_coco_annotations(
            "./dataset/test/_annotations.coco.json",
            test_coco_clean,
            valid_class_ids
        )
        
        register_coco_instances(
            "my_dataset_test", 
            {}, 
            test_coco_clean,
            "./dataset/test/"
        )
    
    return num_classes

class CustomTrainer(DefaultTrainer):
    """
    Custom trainer with evaluation
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def setup_config(num_classes):
    """
    Setup Detectron2 configuration
    """
    cfg = get_cfg()
    
    # Choose a model - Mask R-CNN with ResNet-50 backbone
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # Force CPU mode
    cfg.MODEL.DEVICE = "cpu"

    # Dataset configuration
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # Training configuration
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4  # Adjust based on GPU memory
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000    # Adjust based on dataset size
    cfg.SOLVER.STEPS = (2000,)    # Learning rate decay
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 100
    
    # Evaluation and checkpointing
    cfg.TEST.EVAL_PERIOD = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # Output directory
    cfg.OUTPUT_DIR = "./output_segmentation"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def main():
    """
    Main training function
    """
    print("Registering datasets...")
    num_classes = register_datasets()
    print(f"Found {num_classes} valid classes")
    
    print("Setting up configuration...")
    cfg = setup_config(num_classes)
    
    print("Starting training...")
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()
    
    print("Training completed!")
    print(f"Model saved to: {cfg.OUTPUT_DIR}")
    
    # Optional: Run evaluation on test set if available
    if "my_dataset_test" in DatasetCatalog.list():
        print("Running evaluation on test set...")
        cfg.DATASETS.TEST = ("my_dataset_test",)
        trainer.test(cfg, trainer.model)

def inference_example(model_path, image_path):
    """
    Example inference code
    """
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    import cv2
    
    # Setup config for inference
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Update with your actual number
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    img = cv2.imread(image_path)
    outputs = predictor(img)
    
    # Visualize results
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("my_dataset_train"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save result
    cv2.imwrite("result.jpg", out.get_image()[:, :, ::-1])
    print("Result saved as result.jpg")

if __name__ == "__main__":
    main()
    
    # Example inference (uncomment after training)
    # inference_example("./output_segmentation/model_final.pth", "path/to/test_image.jpg")