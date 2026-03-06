import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

IMG_SIZE=224

classes=["Alzheimer","MCI","Normal"]

svm=joblib.load("svm_model.pkl")
pca=joblib.load("pca.pkl")

base_model=VGG16(weights="imagenet",include_top=False,input_shape=(224,224,3))
vgg=Model(inputs=base_model.input,outputs=base_model.output)

st.title("Alzheimer Early Detection System")

uploaded_file=st.file_uploader("Upload MRI Image",type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes=np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    img=cv2.imdecode(file_bytes,1)

    st.image(img,caption="Uploaded Image",use_column_width=True)

    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img=img/255.0

    img_array=np.expand_dims(img,axis=0)

    features=vgg.predict(img_array)

    features=features.reshape(1,-1)

    features_pca=pca.transform(features)

    prediction=svm.predict(features_pca)
    probability=svm.predict_proba(features_pca)[0]

    st.subheader("Prediction Result")

    st.success(classes[prediction[0]])

    st.subheader("Prediction Confidence")

    for label,prob in zip(classes,probability):

        st.write(label,":",round(prob*100,2),"%")

    fig,ax=plt.subplots()

    ax.bar(classes,probability*100)

    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Confidence")

    st.pyplot(fig)