import cv2
import os
import random
from pathlib import Path

# Đường dẫn tới ảnh và nhãn
image_dir = "datasets/cancer_detection/images"
label_dir = "datasets/cancer_detection/labels"
aug_image_dir = "datasets/cancer_detection_aug/images"
aug_label_dir = "datasets/cancer_detection_aug/labels"

# Tạo thư mục mới cho dữ liệu tăng cường
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Các phép tăng cường dữ liệu
def augment_image(img):
    # Lật ngang
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    # Xoay nhẹ
    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, matrix, (w, h))
    return img

# Xử lý từng ảnh và nhãn
for image_path in Path(image_dir).glob("*.jpg"):
    label_path = Path(label_dir) / f"{image_path.stem}.txt"

    # Đọc ảnh và nhãn
    img = cv2.imread(str(image_path))
    with open(label_path, "r") as f:
        labels = f.readlines()

    # Tăng cường dữ liệu
    for i in range(5):  # Tạo 5 bản sao từ mỗi ảnh
        aug_img = augment_image(img)
        aug_image_path = Path(aug_image_dir) / f"{image_path.stem}_aug_{i}.jpg"
        aug_label_path = Path(aug_label_dir) / f"{image_path.stem}_aug_{i}.txt"

        # Lưu ảnh tăng cường
        cv2.imwrite(str(aug_image_path), aug_img)

        # Sao chép nhãn (giữ nguyên tọa độ YOLO)
        with open(aug_label_path, "w") as f:
            f.writelines(labels)
