import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import tensorflow as tf
from io import BytesIO

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


# Loading Models
model = tf.keras.models.load_model('Custom_model.keras')
yolomodel = YOLO('detect/train2/weights/best.pt')
class_names = ["Bird", "Drone"]

# Sidebar
st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload image", type=["jpg", "jpeg", "png"], help="Upload a Drone or Bird image"
)

run_mobilenet = st.sidebar.button("MobileNet Classification")
run_yolo = st.sidebar.button("YOLO Detection")

st.sidebar.markdown("---")
st.sidebar.info("Developed using MobileNet + YOLOv8")


# Main UI
if uploaded_file:

    col1, col2 = st.columns([1.2, 1])

    # Left --> Image Preview
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save temporarily for YOLO
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp.name)

    # Right --> Results
    with col2:
        st.markdown("### Results Panel")

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
                results = yolomodel.predict(temp.name, conf=0.10, iou=0.25)

            # Annotated image
            annotated_bgr = results[0].plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]
            st.image(annotated_rgb, caption="YOLO Detection", use_container_width=True)

            # Show class + confidence
            detections = results[0].boxes
            st.write("### All Detected Objects")

            if len(detections) == 0:
                st.write("No objects detected.")
            else:
                for box in detections:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    class_name = yolomodel.names[cls_id]

                    st.write(f"• **{class_name}** — {conf:.2f}%")

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
