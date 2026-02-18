import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PostureClassifier(nn.Module):
    def __init__(self, input_size=12, num_classes=2):
        super(PostureClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class PostureDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.samples = []
        self.labels = []
        
        # map category_id to 0/1 (dataset has 1: bad, 2: good based on your JSON)
        # We will map: 2 (good) -> 0, 1 (bad) -> 1
        cat_map = {2: 0, 1: 1}
        
        for ann in data['annotations']:
            keypoints = ann.get('keypoints', [])
            if len(keypoints) == 18: # 6 points * 3 (x, y, v)
                # Extract x, y and normalize (data is already 640x640)
                pts = []
                for i in range(0, 18, 3):
                    pts.append(keypoints[i] / 640.0)
                    pts.append(keypoints[i+1] / 640.0)
                
                self.samples.append(pts)
                self.labels.append(cat_map.get(ann['category_id'], 1))
        
        self.samples = torch.tensor(self.samples, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def train_model(json_path, save_path="posture_classifier.pth"):
    print(f"Loading data from {json_path}...")
    dataset = PostureDataset(json_path)
    
    # Split
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=16, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
    
    model = PostureClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = model(bx)
                    _, pred = torch.max(out.data, 1)
                    correct += (pred == by).sum().item()
            acc = 100 * correct / len(val_idx)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Path to your annotations file
    ANNOTATIONS_PATH = r"C:\Users\Sarthak Dhiman\Downloads\Sitting_Posture.v1i.coco\train\_annotations.coco.json"
    train_model(ANNOTATIONS_PATH)
