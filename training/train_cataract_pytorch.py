import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from pathlib import Path
import time
import copy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# --- Config ---
DATA_DIR = r"D:\Disease Prediction\Dataset\cataract_classification"
MODEL_SAVE_PATH = r"D:\Disease Prediction\cataract_model.pth"
CONFUSION_MATRIX_PATH = r"D:\Disease Prediction\cataract_confusion_matrix.png"
IMG_SIZE = 380 # EfficientNet-B4 Native Resolution
BATCH_SIZE = 8 # Reduced further for memory safety
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 2
CLASS_NAMES = ['Cataract', 'Normal']
PATIENCE = 5 # Early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path=MODEL_SAVE_PATH):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves complete model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        # Save complete model object (not just state_dict) for easier loading
        torch.save(model, self.path)

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

def train_model():
    # --- Transforms ---
    data_transforms = {
        'train': transforms.Compose([
            # Close-up specific: RandomResizedCrop helps with tight framing and distance variations
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            # Macro shots often have varying focus/sharpness
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Datasets ---
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")
    
    # --- WeightedSampler ---
    train_targets = image_datasets['train'].targets
    class_counts = torch.tensor([train_targets.count(i) for i in range(len(class_names))])
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # --- DataLoaders ---
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, num_workers=0),
        'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # --- Model ---
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE, path=MODEL_SAVE_PATH)

    # --- AMP Setup ---
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    since = time.time()
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch}/{NUM_EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            pbar = tqdm(dataloaders[phase], desc=f'{phase} Phase', leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'test':
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                print(f'  Val Metrics: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

                scheduler.step(epoch_loss)
                early_stopping(epoch_loss, model)
                
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
        
        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {early_stopping.best_loss:.4f}')

    # Load best model
    if os.path.exists(MODEL_SAVE_PATH):
        model = torch.load(MODEL_SAVE_PATH)
        model = model.to(device)
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc='Final Evaluation'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(all_labels, all_preds, class_names, CONFUSION_MATRIX_PATH)
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return model

if __name__ == '__main__':
    train_model()
