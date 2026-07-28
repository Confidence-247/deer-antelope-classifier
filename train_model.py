"""
train_model.py
----------------
Trains a binary image classifier (Deer vs Antelope) using transfer learning
with MobileNetV2 as the frozen feature extractor.

Expected folder structure (create this yourself before running):

data/
├── train/
│   ├── deer/       <- put deer training images here (jpg/png)
│   └── antelope/   <- put antelope training images here
└── val/
    ├── deer/       <- put deer validation images here
    └── antelope/   <- put antelope validation images here

Run:
    python train_model.py

Output:
    model/deer_antelope_model.h5   <- the trained model, used by app.py
    model/training_history.png     <- accuracy/loss curves
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODEL_OUT_DIR = "model"
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "deer_antelope_model.h5")

os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 2. DATA GENERATORS (with light augmentation for the training set)
# ---------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["antelope", "deer"],  # antelope=0, deer=1 (alphabetical, explicit for clarity)
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["antelope", "deer"],
)

print("Class indices:", train_generator.class_indices)
# Save class indices so app.py can map prediction -> label reliably
import json
with open(os.path.join(MODEL_OUT_DIR, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f)

# ---------------------------------------------------------------------
# 3. BUILD MODEL (transfer learning)
# ---------------------------------------------------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze the pretrained backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)  # binary classification

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------
# 4. TRAIN
# ---------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT_PATH, monitor="val_accuracy", save_best_only=True),
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------
# 5. OPTIONAL FINE-TUNING (unfreeze last layers for a few more epochs)
# ---------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
)

# ---------------------------------------------------------------------
# 6. SAVE FINAL MODEL
# ---------------------------------------------------------------------
model.save(MODEL_OUT_PATH)
print(f"Model saved to {MODEL_OUT_PATH}")

# ---------------------------------------------------------------------
# 7. PLOT TRAINING CURVES
# ---------------------------------------------------------------------
def combine_hist(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

acc = combine_hist(history, fine_tune_history, "accuracy")
val_acc = combine_hist(history, fine_tune_history, "val_accuracy")
loss = combine_hist(history, fine_tune_history, "loss")
val_loss = combine_hist(history, fine_tune_history, "val_loss")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.legend()
plt.title("Loss")

plt.savefig(os.path.join(MODEL_OUT_DIR, "training_history.png"))
print("Training history plot saved.")
