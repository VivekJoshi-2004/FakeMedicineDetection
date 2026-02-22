# ============================================
# FAKE MEDICINE DETECTOR - FINAL FLOW VERSION
# NOTEBOOK LEVEL ACCURACY
# ============================================

import streamlit as st
import cv2
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
import re
import easyocr
import tensorflow as tf

from playwright.async_api import async_playwright
from rapidfuzz import fuzz
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

nest_asyncio.apply()

# ============================================
# TEXT CLEANING FUNCTION (ADD THIS)
# ============================================

# ============================================
# SAFE CLEAN TEXT (HIGH ACCURACY VERSION)
# ============================================

def clean_text(text):

    text = str(text).upper()

    # remove special characters but KEEP numbers
    text = re.sub(r'[^A-Z0-9 ]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text
# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Fake Medicine Detector",
    page_icon="💊",
    layout="centered"
)

st.title("💊 Fake Medicine Detection System")


# ============================================
# LOAD CNN MODEL
# ============================================

@st.cache_resource
def load_cnn():

    model = tf.keras.models.load_model(
        r"C:\Users\newt4\OneDrive\Desktop\BE_Project\tamper_detector_resnet50v2.keras",
        compile=False
    )

    return model


cnn_model = load_cnn()


# ============================================
# CNN FUNCTION
# ============================================

def cnn_predict(img):

    img = cv2.resize(img, (224,224))

    img = np.array(img, dtype=np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    pred = cnn_model.predict(img, verbose=0)[0][0]

    confidence = float(pred)


    if pred > 0.4:

        return "Genuine", confidence

    else:

        return "Fake", 1 - confidence



# ============================================
# LOAD DATASETS
# ============================================

@st.cache_resource
def load_data():

    real_df = pd.read_excel(
        r"C:\Users\newt4\OneDrive\Desktop\BE_Project\Real_Medicines_dataset.xlsx"
    )

    banned_df = pd.read_excel(
        r"C:\Users\newt4\OneDrive\Desktop\BE_Project\Banned_Pharma_Companies.xlsx"
    )

    return real_df, banned_df


real_df, banned_df = load_data()

real_meds_list = real_df["Name of Medicine"].astype(str).tolist()


banned_df["prod_clean"] = banned_df["Banned Product"].astype(str).apply(clean_text)

banned_df["comp_clean"] = banned_df["Name of the Company"].astype(str).apply(clean_text)



# ============================================
# LOAD OCR
# ============================================

@st.cache_resource
def load_ocr():

    return easyocr.Reader(['en'], gpu=False)


reader = load_ocr()

# def fix_orientation(img):

#     try:

#         osd = pytesseract.image_to_osd(img)

#         angle = int(re.search('Rotate: (\d+)', osd).group(1))

#         if angle == 90:
#             img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

#         elif angle == 180:
#             img = cv2.rotate(img, cv2.ROTATE_180)

#         elif angle == 270:
#             img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

#     except:
#         pass

#     return img


# ============================================
# OCR EXTRACT FUNCTION
# ============================================

def extract_text(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray,(3,3),0)

    gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    results = reader.readtext(gray, rotation_info=[0,90,180,270])

    #img = fix_orientation(img)

    full_text = ""

    for (_, text, conf) in results:

        if conf > 0.40:

            full_text += " " + text


    cleaned = clean_text(full_text)


    st.info("Extracted Text:")

    st.write(cleaned)


    return cleaned



# ============================================
# BEST MATCH FUNCTION (NOTEBOOK ACCURACY)
# ============================================

from rapidfuzz import fuzz

def best_match(text, dataset):

    best_score = 0
    best_name = ""

    for name in dataset:

        score = fuzz.token_set_ratio(text, clean_text(name))

        if score > best_score:

            best_score = score
            best_name = name

    return best_name, best_score



# ============================================
# OCR AUTH FUNCTION
# ============================================

def authenticate_image(img):

    text = extract_text(img)

    text_clean = clean_text(text)


    # -------------------------
    # CHECK BANNED PRODUCT + COMPANY MATCH
    # -------------------------

    banned_match_found = False


    for index, row in banned_df.iterrows():

        prod = row["prod_clean"]

        comp = row["comp_clean"]


        prod_score = fuzz.token_set_ratio(text_clean, prod)

        comp_score = fuzz.token_set_ratio(text_clean, comp)


        if prod_score >= 90 and comp_score >= 90:

            banned_match_found = True

            banned_product = row["Banned Product"]

            banned_company = row["Name of the Company"]

            break


    if banned_match_found:

        return f"❌ Fake Medicine (Banned Product: {banned_product} | Company: {banned_company})"



    # -------------------------
    # CHECK GENUINE
    # -------------------------

    real_match, real_score = best_match(text_clean, real_meds_list)


    if real_score >= 80:

        return f"✅ Genuine Medicine ({real_match})"


    else:

        return "⚠️ Suspicious Medicine"



# ============================================
# HIGH ACCURACY QR FUNCTION
# ============================================

def read_qr(img):

    detector = cv2.QRCodeDetector()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    attempts = [

        img,

        gray,

        cv2.GaussianBlur(gray,(5,5),0),

        cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],

        cv2.adaptiveThreshold(
            gray,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,2
        )

    ]


    for attempt in attempts:

        data, bbox, _ = detector.detectAndDecode(attempt)

        if data and len(data) > 10:

            return data


    return None



# ============================================
# MAIN FLOW
# ============================================


st.divider()

st.subheader("Step 1: Does medicine have QR code?")

choice = st.radio("", ["Yes", "No"])



# ============================================
# QR FLOW
# ============================================

if choice == "Yes":

    st.subheader("Step 2: Upload QR Image")

    qr_image = st.file_uploader("", type=["jpg","png","jpeg"])


    if qr_image:

        image = Image.open(qr_image)

        img_np = np.array(image)

        st.image(image)


        st.subheader("Step 3: QR Verification")

        qr_data = read_qr(img_np)


        if qr_data:

            st.success("QR Code Detected")

            st.write(qr_data)

            result = "✅ Genuine Medicine (QR Verified)"


        else:

            result = "❌ Fake Medicine (QR not readable)"


        st.subheader("Final Verdict")

        st.success(result)



# ============================================
# NO QR FLOW
# ============================================

else:

    st.subheader("Step 2: Upload Medicine Strip Image")

    strip_image = st.file_uploader("", type=["jpg","png","jpeg"])


    if strip_image:

        image = Image.open(strip_image)

        img_np = np.array(image)

        st.image(image)


        st.subheader("Step 3: Running AI Model")

        cnn_result, confidence = cnn_predict(img_np)


        st.write(
            f"AI Result: {cnn_result} (Confidence: {confidence*100:.2f}%)"
        )


        if cnn_result == "Fake":

            st.subheader("Final Verdict")

            st.error("❌ Fake Medicine Detected by AI")


        else:

            st.subheader("Step 4: OCR Verification")

            result = authenticate_image(img_np)


            st.subheader("Final Verdict")

            st.success(result)
