from ultralytics import YOLO

# Load a pretrained YOLO11 instance segmentation model
model = YOLO("yolo11n-seg.pt")  # can also use .yaml if building from scratch

# Train the model
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,       # adjust based on GPU memory
    name="my_segmentation_experiment"
)

# Optionally, validate afterward
metrics = model.val()
print("Box mAP:", metrics.box.map, "| Mask mAP:", metrics.seg.map)
