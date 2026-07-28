"""
make_placeholder_model.py
--------------------------
Generates a lightweight, UNTRAINED placeholder model with the exact same
input/output shape as the real deer_antelope_model.h5 that train_model.py
produces. This lets you test the full pipeline (GitHub push -> Streamlit
Cloud deploy) TODAY, before your real dataset/training is ready.

Its predictions will be meaningless (basically random ~50/50), but app.py
will run correctly against it. Once train_model.py finishes producing the
real model, just overwrite model/deer_antelope_model.h5 with the real one
and redeploy (or just git push — Streamlit Cloud auto-redeploys).

Run:
    python make_placeholder_model.py
"""

import json
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

IMG_SIZE = (224, 224)
MODEL_OUT_DIR = "model"
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "deer_antelope_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_OUT_DIR, "class_indices.json")

os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# Same architecture as train_model.py, but we DON'T train it —
# this is purely to validate save/load + deployment plumbing.
# Try pretrained ImageNet weights first (normal case); if the environment
# has no internet access to download them (e.g. a restricted sandbox),
# fall back to random-initialized weights — still fine for a placeholder,
# since predictions are meaningless either way.
try:
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    print("Loaded MobileNetV2 with pretrained ImageNet weights.")
except Exception as e:
    print(f"Could not download ImageNet weights ({e}). Using random init instead.")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights=None
    )
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.save(MODEL_OUT_PATH)

class_indices = {"antelope": 0, "deer": 1}
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(class_indices, f)

print(f"Placeholder model saved to {MODEL_OUT_PATH}")
print(f"Class indices saved to {CLASS_INDICES_PATH}")
print("This model is UNTRAINED — predictions are meaningless.")
print("Use it only to test the app + deployment pipeline, then replace it")
print("with the real model produced by train_model.py.")
