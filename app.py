import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en', 'id'], gpu=False)

# Set page config
st.set_page_config(page_title="Vehicle Detection", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle Detection App")
st.write("Upload an image to detect vehicles using the custom YOLO model.")

@st.cache_resource
def load_model():
    # Load the custom trained model
    model = YOLO('best.pt')
    return model

with st.spinner("Loading model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Sidebar for controls
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05
)
enable_ocr = st.sidebar.checkbox("Enable OCR (License Plate Recognition)", value=False)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Create two columns to show original and detected side-by-side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with st.spinner('Detecting objects...'):
        # Perform inference. Predict supports PIL Image.
        results = model.predict(image, conf=conf_threshold)
        
        # We get the first result because we only passed one image
        res = results[0]
        
        # Plot the predictions on the image (returns a BGR numpy array)
        res_plotted = res.plot()
        
        # Convert BGR back to RGB for displaying with st.image
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
    with col2:
        st.subheader("Detection Results")
        st.image(res_plotted_rgb, use_container_width=True)
    
    # Show summary of detections
    if len(res.boxes) > 0:
        st.success(f"Successfully detected {len(res.boxes)} object(s).")
        
        if enable_ocr:
            st.divider()
            st.subheader("OCR Results (Detected Text in Vehicles)")
            with st.spinner("Extracting text..."):
                try:
                    reader = load_ocr()
                    img_array = np.array(image.convert("RGB"))
                    
                    found_text = False
                    for i, box in enumerate(res.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        vehicle_crop = img_array[y1:y2, x1:x2]
                        
                        if vehicle_crop.size == 0:
                            continue
                            
                        ocr_results = reader.readtext(vehicle_crop)
                        
                        st.write(f"**Vehicle {i+1}**")
                        col_crop, col_text = st.columns([1, 2])
                        with col_crop:
                            st.image(vehicle_crop, use_container_width=True)
                        with col_text:
                            if not ocr_results:
                                st.write("_No text detected_")
                            else:
                                found_text = True
                                for bbox, text, conf in ocr_results:
                                    st.write(f"- {text} (conf: {conf:.2f})")
                                    
                except Exception as e:
                    st.error(f"Error running OCR: {e}")
                    
    else:
        st.info("No vehicles detected. Try lowering the confidence threshold.")
