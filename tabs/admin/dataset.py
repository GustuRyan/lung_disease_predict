import requests
import streamlit as st
import os
import zipfile
from PIL import Image
from streamlit_cookies_manager import EncryptedCookieManager

API_URL = "http://localhost:8080/api/v1/dataset"
DISEASE_URL = "http://localhost:8080/api/v1/disease"

def submit_dataset(dataset_name, file_path, description):
    
    resp = requests.post(API_URL, json={"user_id": 1, "dataset_name": dataset_name, "file_path": file_path, "description": description})
        
    if resp.status_code == 200:
        st.success("Sukses membuat dataset baru!")
    else:
        st.error(resp.json().get("error", "Failed"))
        
def submit_disease(disease_name, description):
    resp = requests.post(DISEASE_URL, json={"disease_name": disease_name, "description": description})
        
    if resp.status_code == 200:
        st.success("Sukses mendaftarkan penyakit baru!")
    else:
        st.error(resp.json().get("error", "Failed"))    

def dataset_tab():
    st.title("Upload Dataset ke Local System")
    
    resp = requests.get(API_URL)
    datasets = resp.json()
    dataset_options = {d["dataset_name"]: d["ID"] for d in datasets}

    # Input nama folder dataset
    dataset_name = st.selectbox("Pilih Dataset", dataset_options.keys())
    
    new_dataset = st.checkbox("Folder Baru?")
    
    if new_dataset:
        dataset_name = st.text_input("Nama Folder")
        description = st.text_input("Deskripsi Dataset")
        
        new_disease = st.checkbox("Penyakit Baru?")
        
        if new_disease:
            disease_name = st.text_input("Nama Penyakit Baru")
            disease_description = st.text_input("Deskripsi Penyakit baru")
        
    # Upload multiple images
    uploaded_files = st.file_uploader(
        "Upload images (1 or more!)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Upload ZIP folder
    uploaded_zip = st.file_uploader(
        "Upload folder dataset (ZIP)",
        type=["zip"]
    )
    
    if st.button("Save / Extract Dataset"):
        if not dataset_name:
            st.error("Must fill the folder name!")
            return

        if not uploaded_files and not uploaded_zip:
            st.error("Please upload images or ZIP file!")
            return

        save_dir = os.path.join("datasets", dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # --------------------------
        # CASE 1: Extract ZIP FILE
        # --------------------------
        if uploaded_zip is not None:
            with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                zip_ref.extractall(save_dir)
            st.success(f"ZIP extracted to: {save_dir}")
            if new_dataset:
                submit_dataset(dataset_name, f"datasets/{dataset_name}", description)
                if new_disease:
                    submit_disease(disease_name, disease_description)

        # --------------------------
        # CASE 2: Save multiple images
        # --------------------------
        if uploaded_files:
            for file in uploaded_files:
                file_path = os.path.join(save_dir, file.name)

                img = Image.open(file)
                img.save(file_path)

            st.success(f"{len(uploaded_files)} images saved to: {save_dir}")
            if new_dataset:
                submit_dataset(dataset_name, f"datasets/{dataset_name}", description)
                if new_disease:
                    submit_disease(disease_name, disease_description)

        # --------------------------
        # Preview image (max 10)
        # --------------------------
        st.subheader("Preview saved images:")

        image_files = [
            f for f in os.listdir(save_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if len(image_files) == 0:
            st.info("There's no file found.")
        else:
            for img_name in image_files[:10]:
                img_path = os.path.join(save_dir, img_name)
                st.image(img_path, caption=img_name, width=150)