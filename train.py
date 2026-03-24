import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import YOLODataset
from loss_function import LossFunction
from yolov1 import YOLO
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs_training")

def train():
    # Hyperparameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 135

    # Dataset and DataLoader
    dataset = YOLODataset(img_folder="VOC2007/JPEGImages", label_folder="VOC2007/Annotations")
    train_split = int(0.8 * len(dataset))
    train, val = torch.utils.data.random_split(dataset, [train_split, len(dataset) - train_split])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = YOLO()
    criterion = LossFunction()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=5e-4)
    model.to(device)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        if epoch % 50 == 49:  # Print every 50 epochs
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
            model.save(model.state_dict(),"yolov1_epoch_{}.pth".format(epoch+1))
        writer.add_scalar("Loss/Train", total_loss/len(train_loader), epoch)
        writer.add_scalar("Loss/Val", val_loss/len(val_loader), epoch)