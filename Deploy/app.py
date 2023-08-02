import streamlit as st
from PIL import Image
import numpy as np
import cv2
import onnxruntime as ort

HEIGHT, WIDTH = 256, 256

try:
    ort_session_fp16 = ort.InferenceSession("model.fp16.onnx", providers=["CUDAExecutionProvider"])
except:
    ort_session_fp16 = ort.InferenceSession("model.fp16.onnx", providers=["CPUExecutionProvider"])


def preprocess(img):
    h, w = img.shape[0], img.shape[1]
    resized_image = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    image_np = resized_image / 255.0  # Normalize to [0, 1]

    # Define mean and standard deviation for normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Perform normalization using mean and standard deviation
    normalized_image = (image_np - mean) / std
    normalized_image = np.transpose(normalized_image, (2, 0, 1))

    return normalized_image, h, w


def postprocess(predict, h, w):
    a = np.transpose(predict, (1, 2, 0))
    a = cv2.resize(a, (w, h))

    p = np.argmax(a, axis=-1)

    p = np.eye(3)[p]
    green = p[:, :, 1] * 255
    green = np.expand_dims(green, axis=-1)
    red = p[:, :, 2] * 255
    red = np.expand_dims(red, axis=-1)
    backgroud = p[:, :, 1] * 0
    backgroud = np.expand_dims(backgroud, axis=-1)
    predict = np.concatenate([backgroud, green, red], -1).astype(np.uint8)

    return predict

# Streamlit App
# Widget tải lên file ảnh
uploaded_file = st.file_uploader("Chọn hình ảnh...", type=["jpg", "jpeg", "png"])
note = Image.open("note.png")
st.image(note, width=150)

if uploaded_file is not None:
#     # Đọc ảnh từ file tải lên
    img = Image.open(uploaded_file)
    img = np.array(img)

    image, h, w = preprocess(img)
    image = np.expand_dims(image, 0).astype(np.float32)

    predict = ort_session_fp16.run(None, {"input": image})[0][0]
    image_post = postprocess(predict, h, w)
#
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Ảnh đầu vào", use_column_width=True)
    with col2:
        st.image(image_post, caption="Kết quả", channels="BGR", use_column_width=True)
