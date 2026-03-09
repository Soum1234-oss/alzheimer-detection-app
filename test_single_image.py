# =====================================================
# ALZHEIMER'S DISEASE PREDICTION – SINGLE IMAGE TEST
# CNN (VGG16) + PCA + SVM
# =====================================================

# -------------------------
# STEP 0: IMPORT LIBRARIES
# -------------------------
import os 
print("directory",os.getcwd())

import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model


# -------------------------
# STEP 1: LOAD TRAINED MODELS
# -------------------------

# Load PCA model (used during training)
pca = joblib.load(r"c:\Users\souma\Downloads\Desktop\Desktop\python\pca_model.pkl")

# Load trained SVM classifier
svm = joblib.load(r"c:\Users\souma\Downloads\Desktop\Desktop\python\svm_model.pkl")

# Load pretrained CNN for feature extraction
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

cnn_model = Model(
    inputs=base_model.input,
    outputs=base_model.output
)


# -------------------------
# STEP 2: DEFINE CLASS LABELS
# (same order as training)
# -------------------------
class_names = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]


# -------------------------
# STEP 3: LOAD & PREPROCESS IMAGE
# -------------------------
img_path = r"C:\Mini-Project\Alzheimer_Early_Detection\data\raw\Alzheimer\ModerateDemented\\ModerateImpairment (435).jpg"


img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image not found. Check image path.")

# Resize to CNN input size
img = cv2.resize(img, (224, 224))

# Normalize pixel values
img = img / 255.0

# Add batch dimension
img = np.expand_dims(img, axis=0)


# -------------------------
# STEP 4: CNN FEATURE EXTRACTION
# -------------------------
features = cnn_model.predict(img)

# Flatten CNN features
features_flat = features.reshape(1, -1)


# -------------------------
# STEP 5: APPLY PCA
# -------------------------
features_pca = pca.transform(features_flat)


# -------------------------
# STEP 6: SVM PREDICTION
# -------------------------
prediction = svm.predict(features_pca)
predicted_label = class_names[prediction[0]]


# -------------------------
# STEP 7: DISPLAY RESULT
# -------------------------
# -------------------------
# STEP 7: DISPLAY RESULT
# -------------------------

print("======================================")
print(" Alzheimer’s Disease Prediction Result")
print("======================================")

predicted_stage = predicted_label

if predicted_stage == "NonDemented":
    print(" Diagnosis: No Dementia Detected")
elif predicted_stage == "VeryMildDemented":
    print(" Diagnosis: Early Stage Dementia")
elif predicted_stage == "MildDemented":
    print(" Diagnosis: Mild Stage Dementia")
else:
    print(" Diagnosis: Moderate Stage Dementia")

print(" Predicted Stage :", predicted_stage)
print("======================================")
