from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.yaml")

# Train model
model.train(
    data="temp_cancer_config.yaml",
    epochs=200,
    imgsz=640,
    batch=8
)
model_path = "cancer_detection_model.pt"
model.save(model_path)
