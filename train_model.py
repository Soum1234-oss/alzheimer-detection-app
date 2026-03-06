import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

DATASET_PATH = r"C:\Mini-Project\Alzheimer_Early_Detection\data\raw"
IMG_SIZE = 224

classes = {
    "Alzheimer":0,
    "MCI":1,
    "Normal":2
}

X=[]
y=[]

print("Loading images...")

for class_name,label in classes.items():

    class_folder=os.path.join(DATASET_PATH,class_name)

    for root,dirs,files in os.walk(class_folder):

        for file in files:

            if file.lower().endswith((".jpg",".png",".jpeg")):

                img_path=os.path.join(root,file)

                img=cv2.imread(img_path)

                if img is None:
                    continue

                img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
                img=img/255.0

                X.append(img)
                y.append(label)

X=np.array(X,dtype="float32")
y=np.array(y)

print("Total images loaded:",len(X))

if len(X)==0:
    raise ValueError("Dataset not loaded. Check folder structure.")

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Loading VGG16...")

base_model=VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

cnn_model=Model(inputs=base_model.input,outputs=base_model.output)

print("Extracting features...")

train_features=cnn_model.predict(X_train)
test_features=cnn_model.predict(X_test)

train_features=train_features.reshape(train_features.shape[0],-1)
test_features=test_features.reshape(test_features.shape[0],-1)

print("Applying PCA...")

pca=PCA(n_components=300)

train_pca=pca.fit_transform(train_features)
test_pca=pca.transform(test_features)

print("Training SVM...")

svm=SVC(kernel="rbf",probability=True)

svm.fit(train_pca,y_train)

y_pred=svm.predict(test_pca)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

joblib.dump(svm,"svm_model.pkl")
joblib.dump(pca,"pca.pkl")

print("Models saved successfully")