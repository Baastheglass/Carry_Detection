import json
import os
from PIL import Image
import shutil

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    """
    Convert COCO format annotations to YOLO format
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # Create category mapping (COCO category_id -> YOLO index)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

    # Process each image
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Copy image file
        src_image_path = os.path.join(images_dir, image_filename)
        dst_image_path = os.path.join(f"{output_dir}/images", image_filename)
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        
        # Create YOLO label file
        label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(f"{output_dir}/labels", label_filename)
        
        # Find annotations for this image
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        
        with open(label_path, 'w') as label_file:
            for ann in image_annotations:
                # Skip supercategory (category_id = 0)
                if ann['category_id'] == 0:
                    continue
                    
                # Convert COCO bbox to YOLO format
                x, y, width, height = ann['bbox']
                
                # Convert to YOLO format (normalized center coordinates)
                center_x = (x + width / 2) / image_width
                center_y = (y + height / 2) / image_height
                norm_width = width / image_width
                norm_height = height / image_height
                
                # Map COCO category_id to YOLO class index
                class_id = categories[ann['category_id']]

                
                label_file.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

if __name__ == "__main__":
    # Convert your datasets
    convert_coco_to_yolo('dataset/test/_annotations.coco.json', 'dataset/test/', 'yolo_dataset/test')
    convert_coco_to_yolo('dataset/train/_annotations.coco.json', 'dataset/train/', 'yolo_dataset/train')
    convert_coco_to_yolo('dataset/valid/_annotations.coco.json', 'dataset/valid/', 'yolo_dataset/valid')