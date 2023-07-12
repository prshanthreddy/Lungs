import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB7 as PretrainedModel, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from skimage.filters import threshold_otsu
import cv2
import time

X = Y = 224

def preprocess_image(image, predicted_label):
    img= cv2.imread(image.name)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh
    total_area = img_otsu.size
    black_area = np.count_nonzero(img_otsu == 0)
    white_area = np.count_nonzero(img_otsu == 1)
    print(total_area)
    print(white_area)
    print(black_area)
    if predicted_label != 'lung_n':
        if white_area >= 300000:
            level = 3
        elif 200000 <= white_area < 3000000:
            level = 2
        elif  100000 <= white_area < 200000:
            level = 1
        elif white_area < 100000:
            level = 0

        st.warning(f'Predicted as {predicted_label} in Level {level}')
    else:
        st.success("Predicted as Lung with Benign Tumor")

# Create a Streamlit application
def main():
    st.title('Lung Cancer Detection using Deep Learning')
    input_shape = (X, Y, 3)
    K = 3
    start_time = time.time()
    ptm = PretrainedModel(
        input_shape=(X, Y, 3),
        weights='imagenet',
        include_top=False
    )
    ptm.trainable = False
    x = GlobalAveragePooling2D()(ptm.output)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    y = Dense(K, activation='softmax')(x)
    model = Model(inputs=ptm.input, outputs=y)
    model.load_weights('https://lungsbucket.s3.eu-north-1.amazonaws.com/lungs_weights.h5')
    # model.save('lungs_model.h5')

    label_map = {0: 'lung_aca', 1: 'lung_n', 2: 'lung_scc'}

    uploaded_image = st.file_uploader('Upload an image')

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        predicted_label = label_map[predicted_class]
        preprocess_image(uploaded_image, predicted_label)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # st.write('Predicted Class:', predicted_label)
        st.write('Time taken to predict is approximately:', round(elapsed_time, 2), 'seconds')
        st.image(image, caption='Uploaded Image', width=300)
    else:
        st.write('Please upload an image')

if __name__ == '__main__':
    main()
