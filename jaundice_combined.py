# ================================
# 0. BACKEND + GPU CHECK
# ================================
import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "WARNING: Running on CPU")

# ================================
# 1. IMPORTS
# ================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from keras import mixed_precision
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

mixed_precision.set_global_policy("mixed_float16")

# ================================
# 2. CONFIG 
# ================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 20
MODEL_SAVE_PATH = Path.cwd() / "saved_models" / "jaundice_model.keras"

# ================================
# 3. DATA PATH DETECTION
# ================================
workspace_root = Path.cwd()
candidates_jaundice = [
    workspace_root / "Dataset" / "Dataset_Cleaned" / "jaundice"
]

jaundice_dir = next((str(p) for p in candidates_jaundice if p.exists()), None)
if not jaundice_dir:
    print("ERROR: Jaundice dataset not found.")
    sys.exit(1)

path_obj = Path(jaundice_dir)
normal_dir = str(path_obj.parent / ("normal" if path_obj.name == "jaundice" else "normal_processed"))

print(f"Data Found:\n Jaundice: {jaundice_dir}\n Normal: {normal_dir}")

# ================================
# 4. DATAFRAME
# ================================
def list_images(directory):
    return [str(p) for p in Path(directory).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

jaundice_images = list_images(jaundice_dir)
normal_images = list_images(normal_dir)

df = pd.DataFrame({
    "image_path": jaundice_images + normal_images,
    "label": ["Jaundice"] * len(jaundice_images) + ["Normal"] * len(normal_images)
}).sample(frac=1, random_state=42).reset_index(drop=True)

# ================================
# 5. AUGMENTATION + GENERATORS
# ================================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    brightness_range=(0.8, 1.2),
    channel_shift_range=20.0,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_dataframe(
    df, x_col="image_path", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="training"
)

val_gen = datagen.flow_from_dataframe(
    df, x_col="image_path", y_col="label",
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation"
)

# ================================
# 6. CLASS WEIGHTS
# ================================
cls_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(cls_weights))

# ================================
# 7. BUILD MODEL
# ================================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid", dtype="float32")(x)

model = Model(base_model.input, output)

loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.05)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=loss_fn,
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

# ================================
# 8. CALLBACKS
# ================================
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True, monitor="val_auc", mode="max"),
    ModelCheckpoint(str(MODEL_SAVE_PATH), save_best_only=True, monitor="val_auc", mode="max"),
    ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=2, mode='max', min_lr=1e-7)
]

# ================================
# 9. PHASE 1 (FROZEN)
# ================================
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
          callbacks=callbacks, class_weight=class_weights_dict)

# ================================
# 10. SMART FINE-TUNING
# ================================
for layer in base_model.layers[:int(len(base_model.layers)*0.7)]:
    layer.trainable = False
for layer in base_model.layers[int(len(base_model.layers)*0.7):]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=loss_fn,
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

model.fit(train_gen, validation_data=val_gen,
          epochs=EPOCHS + FINE_TUNE_EPOCHS,
          initial_epoch=EPOCHS,
          callbacks=callbacks,
          class_weight=class_weights_dict)

# ================================
# 11. SAVE FINAL MODEL
# ================================
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
