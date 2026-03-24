import os
import random
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from dataset import YOLODataset
from loss_function import LossFunction
from parse_output import parse_yolo_output

def test():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load("yolov1_epoch_135.pth", map_location=device)
    model.to(device)
    CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    dataset = YOLODataset(img_folder="VOC2007/JPEGImages", label_folder="VOC2007/Annotations")
    train_split = int(0.8 * len(dataset))
    train, test = torch.utils.data.random_split(dataset, [train_split, len(dataset) - train_split])
    test_loader = DataLoader(test, batch_size=1, shuffle=True)
    model.eval()
    with torch.no_grad():
        step = 0
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            parsed_boxes = parse_yolo_output(outputs.cpu(), conf_threshold=0.3)
            for i in range(len(parsed_boxes)):
                print(f"Image {i+1}:")
                for box in parsed_boxes[i]:
                    x1, y1, x2, y2, conf, class_id = box
                    print(f"  Class: {CLASSES[int(class_id)]}, Confidence: {conf:.4f}, Box: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})")
                    cv2.rectangle(images.cpu().permute(1, 2, 0).numpy().astype("uint8"), (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(images.cpu().permute(1, 2, 0).numpy().astype("uint8"), f"{CLASSES[int(class_id)]}: {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            cv2.imwrite(f"output/test_output_{step}.jpg", images.cpu().permute(1, 2, 0).numpy().astype("uint8"))
            step += 1