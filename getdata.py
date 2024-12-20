import cv2
import os
import random
from pathlib import Path


image_dir = "datasets/cancer_detection/images"
label_dir = "datasets/cancer_detection/labels"
aug_image_dir = "datasets/cancer_detection_aug/images"
aug_label_dir = "datasets/cancer_detection_aug/labels"


os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)


def augment_image(img):

    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    angle = random.randint(-15, 15)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, matrix, (w, h))
    return img


for image_path in Path(image_dir).glob("*.jpg"):
    label_path = Path(label_dir) / f"{image_path.stem}.txt"

    img = cv2.imread(str(image_path))
    with open(label_path, "r") as f:
        labels = f.readlines()


    for i in range(5):
        aug_img = augment_image(img)
        aug_image_path = Path(aug_image_dir) / f"{image_path.stem}_aug_{i}.jpg"
        aug_label_path = Path(aug_label_dir) / f"{image_path.stem}_aug_{i}.txt"


        cv2.imwrite(str(aug_image_path), aug_img)


        with open(aug_label_path, "w") as f:
            f.writelines(labels)
