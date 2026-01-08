import numpy as np
import requests
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import joblib
import os

from model_cnn import TinyCNN
from model_svm import extract_hog_opencv
from utils.cookie import get_user_id

HISTORY_URL = "http://localhost:8080/api/v1/history"
MODEL_URL = "http://localhost:8080/api/v1/model-active"
DISEASE_URL = "http://localhost:8080/api/v1/disease"
RECOMMENDATION_URL = "http://localhost:8080/api/v1/recommendations"

def dashboard_tab():

    resp = requests.get(MODEL_URL)
    models = resp.json()

    model_options = {d["model_name"]: d["file_path"] for d in models}

    st.title("🔍 Prediksi Menggunakan Model CNN / SVM (.joblib)")

    selected_model_name = st.selectbox(
        "Pilih Model", options=list(model_options.keys())
    )
    
    model_path = model_options[selected_model_name]
    model_data = joblib.load(model_path)

    model_type = model_data.get("model_type")
    class_names = model_data["classes"]

    st.write("Model Type:", model_type.upper())
    st.write("Classes:", class_names)

    uploaded_file = st.file_uploader("Upload Gambar X-Ray", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Input Image", width=300)

    # ==================================================
    # CNN PREDICTION
    # ==================================================
    if model_type == "cnn":
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TinyCNN(num_classes=len(class_names)).to(device)
        model.load_state_dict(model_data["state_dict"])
        model.eval()

        preprocess = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    # ==================================================
    # SVM PREDICTION
    # ==================================================
    elif model_type == "svm":
        svm_model = model_data["model"]

        feat = extract_hog_opencv(img)
        probs = svm_model.predict_proba([feat])[0]

    else:
        st.error("Model tidak dikenali")
        return

    # ==================================================
    # DISPLAY RESULT
    # ==================================================
    pred_idx = int(np.argmax(probs))
    confidence = probs[pred_idx] * 100

    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Prediksi:** {class_names[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(confidence / 100)

    st.subheader("📊 Probabilitas Tiap Class")
    for cls, p in zip(class_names, probs):
        st.write(f"{cls}: {p * 100:.2f}%")
        
    disease_name = class_names[pred_idx].lower()
        
    response = requests.get(f"{DISEASE_URL}/{disease_name}")
    disease = response.json()
    
    st.write("Deskripsi: " + disease["description"])
    
    disease_id = disease["ID"]
    resRecom = requests.get(f"{RECOMMENDATION_URL}/{disease_id}")
    recommendations = resRecom.json()
    
    st.subheader("Rekomendasi Kesehatan") 
    for recom in recommendations:
        st.write(recom["type"] + ": " + recom["recommendation_text"])
    
    user_id = st.session_state.get("user_id")

    if not user_id:
        st.warning("Silakan login terlebih dahulu")
        st.stop()
        
    else:
        save = st.button("Simpan Hasil Prediksi")
    
        if save:
            save_dir = f"uploads\{user_id}"
            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(save_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File berhasil disimpan di {file_path}")
            
            saveHistory = requests.post(HISTORY_URL, json={"user_id": user_id, "disease_id": disease_id, "prediction_result": disease_name, "confidence_result": f"{confidence:.2f}%", "image_path": f"uploads/{user_id}/{uploaded_file.name}",})
            
            saveHistory.raise_for_status()