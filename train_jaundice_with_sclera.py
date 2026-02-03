import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras.utils import Sequence
from keras.applications import EfficientNetB0
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate, Conv2D, MaxPool2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Config
# Pointing directly to the folder you created
DATASET_ROOT = Path(r"D:\Disease Prediction\Dataset\combined_sclera")
IMG_DIR = DATASET_ROOT / "JPEGImages"
MASK_DIR = DATASET_ROOT / "SegmentationClass"

def get_label(filename):
    """
    Infers label from filename.
    Jaundice files usually have 'jaundice' in the name.
    Normal files are '10001', 'plain', 'kaggle', etc.
    """
    name = filename.lower()
    if 'jaundice' in name:
        return 1 # Jaundice
    return 0 # Normal

class DirectDataGenerator(Sequence):
    """
    Generates data directly from the combined_sclera folder using Images and Masks.
    Does NOT use a CSV. Performs on-the-fly cropping.
    """
    def __init__(self, image_files, batch_size=16, img_size=(224,224), sclera_size=(64,64), shuffle=True):
        self.image_files = image_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.sclera_size = sclera_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.image_files))
        batch_idx = self.indexes[start:end]
        
        batch_files = [self.image_files[i] for i in batch_idx]
        
        skins = []   # Original Image
        scleras = [] # Cropped Sclera
        labels = []
        
        for img_path in batch_files:
            # 1. Load Original Image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Label
            label = get_label(img_path.name)
            
            # 2. Load Mask & Crop Sclera
            stem = img_path.stem
            mask_path = MASK_DIR / f"{stem}.png"
            
            sclera_crop = np.zeros((*self.sclera_size, 3), dtype=np.float32)
            
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) # Load mask
                
                # Handle resizing if mask/img mismatch
                if mask is not None:
                     # Check if mask is 3D (H,W,3) or 2D (H,W)
                    if len(mask.shape) == 3:
                        mask = mask[:,:,0] # Take one channel
                        
                    if mask.shape != img.shape[:2]:
                         mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert mask to binary (anything > 0 is sclera)
                    # Note: You used RGB values [0, 81, 100, 230], so >0 works.
                    binary_mask = (mask > 0).astype(np.uint8)
                    
                    # Find contours or bounding box
                    points = cv2.findNonZero(binary_mask)
                    if points is not None:
                        x, y, w, h = cv2.boundingRect(points)
                        
                        # Apply mask to image (black background)
                        masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
                        
                        # Crop
                        sclera_crop_raw = masked_img[y:y+h, x:x+w]
                        
                        if sclera_crop_raw.size > 0:
                            # Resize to 64x64 input
                            sclera_crop = cv2.resize(sclera_crop_raw, self.sclera_size)
                            sclera_crop = sclera_crop.astype('float32') / 255.0
            
            # Preprocess Main Image
            img_resized = cv2.resize(img, self.img_size)
            img_resized = img_resized.astype('float32') / 255.0
            
            skins.append(img_resized)
            scleras.append(sclera_crop)
            labels.append(label)
            
        return (
            {'skin_input': np.array(skins), 'sclera_input': np.array(scleras)},
            np.array(labels)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def make_model(input_shape=(224,224,3), sclera_shape=(64,64,3)):
    # Standard Architecture
    # 1. Skin Branch (EfficientNet)
    skin_input = Input(shape=input_shape, name='skin_input')
    base = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=skin_input)
    x1 = GlobalAveragePooling2D()(base.output)

    # 2. Sclera Branch (Custom CNN)
    sclera_input = Input(shape=sclera_shape, name='sclera_input')
    y = Conv2D(32, 3, activation='relu', padding='same')(sclera_input)
    y = MaxPool2D()(y)
    y = Conv2D(64, 3, activation='relu', padding='same')(y)
    y = GlobalAveragePooling2D()(y)

    # 3. Fusion
    merged = Concatenate()([x1, y])
    merged = Dropout(0.4)(merged) # Increased dropout
    merged = Dense(64, activation='relu')(merged)
    out = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=[skin_input, sclera_input], outputs=out)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print(f"Loading data from {DATASET_ROOT}...")
    
    # 1. Gather all images
    all_images = list(IMG_DIR.glob("*"))
    all_images = [f for f in all_images if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    if len(all_images) == 0:
        print("No images found! Check path.")
        return

    print(f"Found {len(all_images)} images.")
    
    # 2. Check Class Balance
    labels = [get_label(f.name) for f in all_images]
    jaundice_count = sum(labels)
    normal_count = len(labels) - jaundice_count
    print(f"Balance: Jaundice={jaundice_count}, Normal={normal_count}")
    
    # 3. Split
    train_files, val_files = train_test_split(all_images, test_size=0.2, random_state=42, stratify=labels)
    
    # 4. Generators
    train_gen = DirectDataGenerator(train_files, batch_size=8) # Lower batch size due to processing
    val_gen = DirectDataGenerator(val_files, batch_size=8, shuffle=False)
    
    # 5. Train
    print("Building model...")
    model = make_model()
    
    print("Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15
    )
    
    save_path = Path('saved_models/jaundice_with_sclera.keras')
    save_path.parent.mkdir(exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
