from ultralytics import YOLO
from torchvision.models import densenet121
import torch
import torch.nn as nn

# DenseNet Backbone
class DenseNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = densenet121(pretrained=True)
        self.features = nn.Sequential(*list(densenet.features.children()))  # Lấy các tầng đặc trưng

    def forward(self, x):
        return self.features(x)

# Custom YOLO Model
class CustomYOLOModel(YOLO):
    def __init__(self, model_yaml):
        super().__init__(model_yaml)
        # Thay thế backbone của YOLO
        self.model.model[0] = nn.Sequential(DenseNetBackbone())  # Thay thế module đầu tiên bằng DenseNet

# Load cấu hình YOLO với DenseNet backbone
model = CustomYOLOModel("yolov8n.yaml")

# Huấn luyện mô hình
model.train(
    data="temp_cancer_config.yaml",  # File cấu hình dữ liệu
    epochs=200,                      # Số lượng epochs
    imgsz=640,                       # Kích thước ảnh
    batch=8                          # Batch size
)

# Lưu mô hình sau huấn luyện
model_path = "cancer_detection_densenet_model.pt"
model.save(model_path)