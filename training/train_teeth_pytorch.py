import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from pathlib import Path
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import json

# --- Config ---
DATA_DIR = r"D:\Disease Prediction\Dataset\teeth\train"
TEST_DIR = r"D:\Disease Prediction\Dataset\teeth\test"
MODEL_SAVE_PATH = r"D:\Disease Prediction\saved_models\teeth_model.pth"
MAPPING_SAVE_PATH = r"D:\Disease Prediction\saved_models\teeth_disease_mapping.json"
CONFUSION_MATRIX_PATH = r"D:\Disease Prediction\saved_models\teeth_confusion_matrix.png"
IMG_SIZE = 380  # EfficientNet-B4 Native Resolution
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 7

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
        torch.save(model, self.path)

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Teeth Disease Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

class TeethDiseaseModel(nn.Module):
    """Multi-class classification for teeth pathologies"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def train_model():
    # --- Transforms ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            # Teeth-specific augmentations
            transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Datasets ---
    # Check if test directory exists, otherwise use train for both
    if os.path.exists(TEST_DIR):
        image_datasets = {
            'train': datasets.ImageFolder(DATA_DIR, data_transforms['train']),
            'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
        }
    else:
        print("Warning: Test directory not found. Using train directory for validation.")
        full_dataset = datasets.ImageFolder(DATA_DIR, data_transforms['train'])
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        image_datasets = {'train': train_dataset, 'test': test_dataset}
    
    class_names = image_datasets['train'].classes if hasattr(image_datasets['train'], 'classes') else image_datasets['train'].dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    
    # Save class mapping
    class_to_idx = image_datasets['train'].class_to_idx if hasattr(image_datasets['train'], 'class_to_idx') else image_datasets['train'].dataset.class_to_idx
    with open(MAPPING_SAVE_PATH, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Class mapping saved to {MAPPING_SAVE_PATH}")
    
    # --- WeightedSampler ---
    train_targets = image_datasets['train'].targets if hasattr(image_datasets['train'], 'targets') else [s[1] for s in image_datasets['train']]
    class_counts = torch.tensor([train_targets.count(i) for i in range(num_classes)])
    print(f"Class distribution: {dict(zip(class_names, class_counts.tolist()))}")
    
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
    model = TeethDiseaseModel(num_classes=num_classes).to(device)

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
                
                # Metrics
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

    # Confusion Matrix
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
