# Import necessary libraries
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_extras.add_vertical_space import add_vertical_space
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av


# Configure Streamlit page layout
st.set_page_config(layout="centered")
st.title(":blue[Plant Disease] :red[Detector]")
# st.markdown("#### Empowering farmers with a timely and accurate tool for identifying plant diseases in their crops using Artificial Intelliegence")

with st.sidebar:
    st.header("⚙️**Setup**")
    # Set the number of columns for image display
    n_cols = st.slider("Set number of grid columns", 2, 5)
    model_thresh = st.number_input(
        "Adjust classification model confidence threshold", 0.3, 1.0, 0.5
    )

    object_detection_thresh = st.number_input("Adjust segmentation model confidence threshold", 0.2, 1.0, 0.5)


# Define class names for plant diseases
class_names = [
    "Apple_Apple_scab",
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Apple_healthy",
    "Blueberry_healthy",
    "Cherry_(including_sour)_Powdery_mildew",
    "Cherry_(including_sour)_healthy",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)_Common_rust_",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Corn_(maize)_healthy",
    "Grape_Black_rot",
    "Grape_Esca_(Black_Measles)",
    "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape_healthy",
    "Orange_Haunglongbing_(Citrus_greening)",
    "Peach_Bacterial_spot",
    "Peach_healthy",
    "Pepper,_bell_Bacterial_spot",
    "Pepper,_bell_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_Powdery_mildew",
    "Strawberry_Leaf_scorch",
    "Strawberry_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]

add_vertical_space(3)
tab1, tab2 = st.tabs(["Plant Disaese Classification", "Plant Disease Segmentation"])

with tab1:
    # Define a function to load the pre-trained model
    @st.cache_resource
    def load_models():
        keras_model = tf.keras.models.load_model("Models/saved_model")
        yolo_model = YOLO("Models/best.pt")
        return keras_model, yolo_model

    keras_model, yolo_model = load_models()  # load model

    # Define a function to preprocess an image
    def preprocess_img(img):
        img = tf.keras.utils.load_img(img, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        batched_img = tf.expand_dims(img_array, 0)
        predictions = keras_model.predict(batched_img)
        score = tf.nn.softmax(predictions[0])

        return img_array, score

    # Display a subheading for camera inference
    st.subheader(":blue[📸 Camera Inference]")

    # Capture an image from the camera
    img_file_buffer = st.camera_input("Take a picture of the plant")

    if img_file_buffer is not None:
        # Process the captured image
        bytes_data = img_file_buffer.getvalue()
        img_tensor = tf.io.decode_image(bytes_data, channels=3)
        img_float = tf.image.convert_image_dtype(img_tensor, tf.float32)
        resized_img = tf.image.resize(img_float, (224, 224))
        batched_img = tf.expand_dims(resized_img, 0)
        predictions = keras_model.predict(batched_img)
        score = tf.nn.softmax(predictions[0])

        if np.max(score) < model_thresh:
            st.warning("I'm not very certain about what plant disease it is")
        else:
            # Display the prediction result
            st.write(
                f":orange[Predicted Label]: {class_names[np.argmax(score)]} with {100 * np.max(score) :.2f}% confidence"
            )

    add_vertical_space(3)
    # Display a subheading for batch prediction
    st.subheader(":blue[🗃️Upload Plant Images]")

    # Upload one or more plant images
    uploaded_img = st.file_uploader(
        "Upload an image of a plant", accept_multiple_files=True
    )

    # Button to trigger prediction
    pred_btn = st.button("Predict Crop Disease")
    st.markdown("---")

    if pred_btn:
        if uploaded_img:
            crop_image, pred_label, confidence = [], [], []
            for file in uploaded_img:
                img, score = preprocess_img(file)
                crop_image.append(img.astype("uint8"))
                if np.max(score) < model_thresh:
                    pred_label.append("Not very certain")
                    confidence.append(f"{100 * np.max(score) :.2f}%")
                else:
                    pred_label.append(class_names[np.argmax(score)])
                    confidence.append(f"{100 * np.max(score) :.2f}%")

            n_rows = int(1 + len(crop_image) // n_cols)
            rows = [st.columns(n_cols) for _ in range(n_cols)]
            cols = [column for row in rows for column in row]

            # Display predictions for uploaded images
            for col, poster, p_label, conf in zip(
                cols, crop_image, pred_label, confidence
            ):
                col.markdown(f"###### :orange[Predicted_Disease]: :green[{p_label}]")
                col.markdown(f"###### :orange[Confidence]: :green[{conf}]")
                col.image(poster)
        else:
            st.warning("⚠️You must upload an image before making an inference🤭")


# Instance segmentation
with tab2:
    add_vertical_space(2)
    st.title("Instance Segmentation")
    add_vertical_space(2)

    class VideoProcessor:
        def recv(self, frame):
            frame = frame.to_ndarray(format="bgr24")
            results = yolo_model(frame, conf=object_detection_thresh)  # get results
            frame = results[0].plot()

            return av.VideoFrame.from_ndarray(frame, format="bgr24")

    webrtc_streamer(
        key="stream",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]},
    )
