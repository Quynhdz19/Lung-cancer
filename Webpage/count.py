import os

def count_files(txt_folder, image_folder):
    # Kiểm tra xem các thư mục có tồn tại không
    if not os.path.exists(txt_folder):
        print(f"Thư mục chứa file txt ({txt_folder}) không tồn tại.")
        return 0, 0
    if not os.path.exists(image_folder):
        print(f"Thư mục chứa ảnh ({image_folder}) không tồn tại.")
        return 0, 0

    # Đếm số file txt
    txt_count = len([file for file in os.listdir(txt_folder) if file.endswith('.txt')])

    # Đếm số file ảnh
    image_count = len([file for file in os.listdir(image_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])

    return txt_count, image_count

# Đường dẫn tới các thư mục
# Đường dẫn tới các thư mục
txt_folder_path = "/Volumes/SSD SanDisk/projects/abc/Webpage/data_collection"  # Thay bằng đường dẫn thực tế tới thư mục chứa file txt
image_folder_path = "/Volumes/SSD SanDisk/projects/dataset05/Cancerous"  # Thay bằng đường dẫn thực tế tới thư mục chứa ảnh


# Thực hiện đếm số file
txt_count, image_count = count_files(txt_folder_path, image_folder_path)
print(f"Số file .txt: {txt_count}")
print(f"Số file ảnh: {image_count}")


