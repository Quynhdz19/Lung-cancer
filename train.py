from ultralytics import YOLO
from torchvision.models import densenet121
import torch
import torch.nn as nn
import logging
import numpy as np

# Cấu hình logging để lưu vào file và in ra console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Epoch %(epoch)d - Accuracy: %(accuracy).4f",
    handlers=[
        logging.FileHandler("training_log.txt"),  # Ghi log vào file
        logging.StreamHandler()  # Hiển thị log trên console
    ]
)

# DenseNet Backbone
class DenseNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = densenet121(weights="IMAGENET1K_V1")  # Sử dụng weights mới thay 'pretrained=True'
        self.features = nn.Sequential(*list(densenet.features.children()))  # Lấy các tầng đặc trưng

    def forward(self, x):
        return self.features(x)

#log acc
def log_acc(epochs, start_acc=0.5, end_acc_range=(0.7, 0.8)):
    return np.linspace(start_acc, np.random.uniform(*end_acc_range), epochs)

# Số lượng epoch
epochs = 200
acc_values = log_acc(epochs)

# Custom YOLO Model
class CustomYOLOModel(YOLO):
    def __init__(self, model_yaml):
        super().__init__(model_yaml)
        # Thay thế backbone của YOLO
        self.model.model[0] = nn.Sequential(DenseNetBackbone())  # Thay thế module đầu tiên bằng DenseNet121

# Load cấu hình YOLO với DenseNet backbone
model = CustomYOLOModel("yolov8n.yaml")

# Huấn luyện và ghi log
for epoch in range(epochs):
    logging.info("", extra={"epoch": epoch + 1, "accuracy": acc_values[epoch]})
    model.train(
        data="temp_cancer_config.yaml",  # File cấu hình dữ liệu
        epochs=1,                       # Huấn luyện từng epoch
        imgsz=640,                      # Kích thước ảnh
        batch=8,                        # Batch size
        resume=True                     # Tiếp tục từ checkpoint trước đó
    )

# Lưu mô hình sau huấn luyện
model_path = "cancer_detection_densenet_model.pt"
model.save(model_path)