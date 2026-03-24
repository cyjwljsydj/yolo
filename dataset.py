import os

import torch
import torchvision
import xmltodict
import cv2
from PIL import Image
from torchvision import transforms

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, label_folder, transform=None, S=7, B=2, C=20):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.img_names = os.listdir(img_folder)
        self.label_names = os.listdir(label_folder)

        self.classes_list = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                             "train", "tvmonitor"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        label_name = img_name.replace(".jpg", ".xml")
        label_path = os.path.join(self.label_folder, label_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        # print(f"Original image dimensions: {original_h} x {original_w}")
        img = cv2.resize(img, (448, 448))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        if self.transform:
            img = self.transform(img)
        with open(label_path, "r") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict["annotation"]["object"]
        # if there is only one box in the image, then the return args will be a dictionary type, and we should normalize
        if isinstance(objects, dict):
            objects = [objects]
        
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        occupied_cells = set()  # To track which grid cells are already occupied by a bounding box
        for object in objects:
            label_name = object["name"]
            labels = self.classes_list.index(label_name)
            bbox = object["bndbox"]
            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])
            # yolo vanilla label format: (class, x_center, y_center, width, height) normalized by the original image dimensions
            cx = (xmin + xmax) / 2 / original_w
            cy = (ymin + ymax) / 2 / original_h
            dw = (xmax - xmin) / original_w
            dh = (ymax - ymin) / original_h
            # target format: (class, x_center, y_center, width, height) for each box, and the rest of the grid cells will be 0
            grid_x = int(cx * self.S)
            grid_y = int(cy * self.S)
            if grid_x >= self.S:
                grid_x = self.S - 1
            if grid_y >= self.S:
                grid_y = self.S - 1
            grid_cell = (grid_y, grid_x)
            if grid_cell in occupied_cells: 
                # print(f"Warning: Grid cell ({grid_y}, {grid_x}) is already occupied by another bounding box. Skipping this box.")
                target_x = cx * self.S - grid_x
                target_y = cy * self.S - grid_y
                target[grid_y, grid_x, 5] = 1  # objectness score for the second box (not used in this case)
                target[grid_y, grid_x, 6:10] = torch.tensor([target_x, target_y, dw, dh])  # same as the first box
                target[grid_y, grid_x, 10 + labels] = 1  # one-hot encoding for class label
            else:
                occupied_cells.add(grid_cell)
                target_x = cx * self.S - grid_x
                target_y = cy * self.S - grid_y
                target[grid_y, grid_x, 0] = 1  # objectness score for the first box
                target[grid_y, grid_x, 1:5] = torch.tensor([target_x, target_y, dw, dh])
                target[grid_y, grid_x, 5:10] = torch.tensor([0, 0, 0, 0, 0])  # objectness score for the second box (not used in this case)
                target[grid_y, grid_x, 10 + labels] = 1  # one-hot encoding for class label
            
        return img, target

if __name__ == "__main__":
    img_folder = "VOC2007/JPEGImages"
    label_folder = "VOC2007/Annotations"
    dataset = YOLODataset(img_folder, label_folder)
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target tensor: {target}")
    # cv2.imshow("Image", img.permute(1, 2, 0).numpy().astype("uint8"))
    # cv2.waitKey(0)