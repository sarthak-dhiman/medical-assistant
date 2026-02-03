import os
# Force Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import torch
if torch.cuda.is_available():
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU")

import json
import numpy as np
import keras
from tensorflow.keras import mixed_precision

# Enable Mixed Precision for RTX GPU (Speedup + Less VRAM)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"Mixed Precision Policy: {policy.name}")
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Explicitly import the preprocessing function
from keras.applications.efficientnet import EfficientNetB0, preprocess_input 
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10
EPOCHS_FINE_TUNE = 30  # Increased for deep learning convergence
MODEL_SAVE_PATH = Path.cwd() / "saved_models" / "optimized_skin_model.keras"
MAPPING_SAVE_PATH = Path.cwd() / "saved_models" / "class_indices.json"

# --- 1. DATASET DISCOVERY ---
workspace_root = Path.cwd()
dataset_skin_root = workspace_root / "Dataset" / "skin"
train_dir = dataset_skin_root / "train"
test_dir = dataset_skin_root / "test"

if not train_dir.exists():
    print(f"ERROR: Training folder not found at {train_dir}")
    exit(1)

# --- 2. DATA GENERATORS (CRITICAL FIXES) ---
# FIX: Added preprocessing_function to match EfficientNet's requirements
# FIX: Added brightness_range to handle lighting conditions
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    horizontal_flip=True,
    rotation_range=20,      # Reduced from 40 to preserve lesion shape
    zoom_range=0.2,         # Reduced from 0.3
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2], # Handle dark/bright photos
    fill_mode='nearest',
)

# Test data must ONLY be preprocessed, not augmented
datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

print("Scanning dataset...")
train_gen = datagen_train.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' if not test_dir.exists() else None,
    shuffle=True
)

if test_dir.exists():
    val_gen = datagen_test.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=list(train_gen.class_indices.keys()), # Ensure consistent class mapping
        shuffle=False
    )
else:
    val_gen = datagen_train.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=list(train_gen.class_indices.keys()), # Ensure consistent class mapping
        shuffle=False
    )

num_classes = train_gen.num_classes
class_indices = train_gen.class_indices

# --- CLASS WEIGHTS ---
print("Computing Class Weights...")
cls_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(cls_weights))
# Handle missing classes in training weights (rare but possible with subsetting)
# Ensure dictionary covers all 0..num_classes-1
for i in range(num_classes):
    if i not in class_weights_dict:
        class_weights_dict[i] = 1.0 # Default weight
        
print(f"Weights computed for {len(class_weights_dict)} classes.")

# Save mapping
MAPPING_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MAPPING_SAVE_PATH, 'w') as f:
    json.dump(class_indices, f, indent=4)

# --- 3. IMPROVED MODEL ARCHITECTURE ---
def build_model():
    print("Building Optimized EfficientNetB0...")
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # FIX: Added a Dense learning block
    # This helps the model learn combinations of features before classifying
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x) 
    
    # Final Output (Must be float32 for Mixed Precision stability)
    output = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=base_model.input, outputs=output)
    
    # FIX: Added Label Smoothing to Loss
    # Prevents model from being "too confident" on noisy data
    model.compile(
        optimizer='adam',
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0), # Removed smoothing for sharper decision boundaries
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    return model, base_model

# Check for existing model
RESUMING = False
if MODEL_SAVE_PATH.exists():
    print(f"Found existing model at {MODEL_SAVE_PATH}")
    try:
        loaded_model = keras.models.load_model(str(MODEL_SAVE_PATH))
        
        # Check compatibility
        if loaded_model.output_shape[-1] == num_classes:
            print("Model topology matches. Resuming...")
            model = loaded_model
            # Recompile to ensure optimizer state is fresh if needed
            model.compile(
                optimizer=keras.optimizers.Adam(1e-4), # Increased from 1e-5
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
                metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
            )
            RESUMING = True
            base_model = model # Treat whole model as base when resuming
        else:
            print(f"Dimension Mismatch: Saved model has {loaded_model.output_shape[-1]} classes, but new dataset has {num_classes}.")
            print("DISCARDING old model. Building FRESH model.")
            model, base_model = build_model()
            
    except Exception as e:
        print(f"Error loading model: {e}. Building FRESH model.")
        model, base_model = build_model()
else:
    model, base_model = build_model()

# --- 4. CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(MODEL_SAVE_PATH), monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-7)
]

# --- 5. PHASE 1: FROZEN TRAINING ---
if not RESUMING:
    print(f"\nPHASE 1: Training Head Layers ({EPOCHS_FROZEN} Epochs)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

# --- 6. PHASE 2: INTELLIGENT FINE-TUNING ---
print("\nPHASE 2: Fine-Tuning Top Layers...")

# FIX: Unfreeze ENTIRE model for deep adaptation
# The skin disease dataset is complex enough to warrant full fine-tuning
base_model.trainable = True
print("Full Model Unfrozen for Fine-Tuning.")

# Recompile with LOW Learning Rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),  # Increased from 1e-5 for better convergence
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

total_epochs = EPOCHS_FROZEN + EPOCHS_FINE_TUNE
initial_epoch = EPOCHS_FROZEN if not RESUMING else 0

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=total_epochs if not RESUMING else EPOCHS_FINE_TUNE,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# --- 7. SAVE ---
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")