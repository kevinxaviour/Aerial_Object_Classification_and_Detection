import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import tensorflow as tf
from io import BytesIO
import boto3
import os


# Page Configuration
st.set_page_config(
    page_title="Drone vs Bird | Detection App",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#3b82f6;'>Drone vs Bird Detection App</h1>
    <h4 style='text-align:center; color:gray;'>MobileNet Classification + YOLO Object Detection</h4>
    <br>
    """,
    unsafe_allow_html=True
)
bucket_name = "forestclassification"  
AWS_ACCESS_KEY_ID= os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY= os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION=  os.getenv("AWS_DEFAULT_REGION")
MOBILENET_MODEL_KEY=os.getenv("MOBILENET_MODEL_KEY")
YOLO_MODEL_KEY=os.getenv("YOLO_MODEL_KEY")



s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

@st.cache_resource
def load_all_from_s3():

    mobilenet_obj = s3.get_object(Bucket=bucket_name, Key=MOBILENET_MODEL_KEY)

    mobilenet_bytes = mobilenet_obj["Body"].read()

    mobilenet_path = "/tmp/mobilenet_model.keras"
    with open(mobilenet_path, "wb") as f:
        f.write(mobilenet_bytes)

    mobilenet_model = tf.keras.models.load_model(mobilenet_path)


    yolo_obj = s3.get_object(Bucket=bucket_name, Key=YOLO_MODEL_KEY)

    yolo_bytes = yolo_obj["Body"].read()

    yolo_path = "/tmp/yolo_model.pt"
    with open(yolo_path, "wb") as f:
        f.write(yolo_bytes)

    yolo_model = YOLO(yolo_path)

    return mobilenet_model, yolo_model

model, yolomodel = load_all_from_s3()
# # Loading Models
# model = tf.keras.models.load_model('Custom_model.keras')
# yolomodel = YOLO('detect/train2/weights/best.pt')
class_names = ["Bird", "Drone"]

# Sidebar
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload image", type=["jpg", "jpeg", "png"], help="Upload an image"
)

run_mobilenet = st.sidebar.button("Run MobileNet")
run_yolo = st.sidebar.button("Run YOLO Detection")

st.sidebar.markdown("---")
# st.sidebar.info("Developed using MobileNet + YOLOv8")


if uploaded_file:

    col1, col2 = st.columns([1.2, 1])

    # Left Side -> Image Preview
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Saving Temporarily for YOLO
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp.name)

    # Right Side -> Results Preview
    with col2:
        st.markdown("### 🔍 Results Panel")

        # MobileNet Prediction
        if run_mobilenet:
            with st.spinner("Running MobileNet Prediction..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)
                predicted_class = 1 if preds[0][0] >= 0.5 else 0
                predicted_label = class_names[predicted_class]

            st.success(f"**MobileNet Prediction:** {predicted_label}")

        # YOLO Detection
        if run_yolo:
            with st.spinner("Running YOLO Detection..."):
                results = yolomodel.predict(temp.name, conf=0.25)

            annotated_bgr = results[0].plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]

            st.image(annotated_rgb, caption="YOLO Detection", use_container_width=True)

            # Download button
            pil_img = Image.fromarray(annotated_rgb)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download YOLO Result",
                data=byte_im,
                file_name="YOLO_detection.png",
                mime="image/png"
            )

else:
    st.info("Upload an image from the **sidebar** to begin.")




