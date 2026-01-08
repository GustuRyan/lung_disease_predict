import joblib
import requests
import streamlit as st
import os
import time

import torch

from model_cnn import train_dynamic_classes
from model_svm import train_svm_dynamic_classes

DATASET_URL = "http://localhost:8080/api/v1/dataset"
MODEL_URL = "http://localhost:8080/api/v1/model"
DATASET_MODEL_URL = "http://localhost:8080/api/v1/dataset-model"

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total if total > 0 else 0


def training_tab():

    resp = requests.get(DATASET_URL)
    datasets = resp.json()
    dataset_options = {d["dataset_name"]: d["ID"] for d in datasets}

    st.header("Train Model dari Dataset Lokal (CNN / SVM)")

    dataset_root = "datasets"

    # ==========================
    # LOAD CLASSES
    # ==========================
    all_classes = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    selected_classes = st.multiselect(
        "Pilih Class yang akan digunakan untuk Training",
        all_classes,
        default=all_classes[:2],
    )

    selected_ids = [
        dataset_options[cls] for cls in selected_classes if cls in dataset_options
    ]

    model_type = st.radio("Pilih Model", ["CNN", "SVM (HOG)"])

    model_name = st.text_input("Nama file model (tanpa ekstensi)", "model_flexible")

    epochs = st.slider("Jumlah Epoch", 1, 50, 10)
    batch_size = st.slider("Batch Size", 8, 64, 32)

    train_btn = st.button("Mulai Training")

    if not train_btn:
        return

    if len(selected_classes) < 2:
        st.error("Minimal pilih 2 kelas!")
        return

    st.success(f"Kelas digunakan: {selected_classes}")
    st.info(f"Model dipilih: {model_type}")

    progress = st.progress(0)
    status_area = st.empty()

    # ==========================
    # TRAINING
    # ==========================
    with st.spinner("🔄 Training sedang berjalan..."):

        if model_type == "CNN":
            history, model, class_names = train_dynamic_classes(
                selected_classes,
                dataset_root,
                epochs,
                batch_size,
                progress_callback=lambda e, t: progress.progress((e + 1) / t),
                status_callback=lambda msg: status_area.write(msg),
            )

            train_acc = history["train_acc"][-1] if history["train_acc"] else None
            final_acc = history["val_acc"][-1] if history["val_acc"] else None

            save_path = f"models/cnn/{model_name}.joblib"
            joblib.dump(
                {
                    "model_type": "cnn",
                    "state_dict": model.state_dict(),
                    "classes": class_names,
                },
                save_path,
            )

        else:  # ================= SVM =================

            history, model, class_names = train_svm_dynamic_classes(
                selected_classes,
                dataset_root,
                progress_callback=lambda p: progress.progress(p),
                status_callback=lambda msg: status_area.write(msg),
            )

            train_acc = history["train_acc"]
            final_acc = history["test_acc"]

            save_path = f"models/svm/{model_name}.joblib"
            joblib.dump(
                {"model_type": "svm", "model": model, "classes": class_names}, save_path
            )

    # ==========================
    # RESULT
    # ==========================
    st.success("🎉 Training selesai!")

    st.write("📊 **Hasil Evaluasi Model**")
    st.write(f"Model Type: **{model_type}**")

    if train_acc is not None:
        st.metric("Train Accuracy", f"{train_acc * 100:.2f}%")

    if final_acc is not None:
        st.metric("Validation Accuracy", f"{final_acc * 100:.2f}%")

    st.write("💾 Model disimpan sebagai:", save_path)
    
    respModels = requests.post(MODEL_URL, json={"model_name": model_name, "file_path": save_path, "accuracy": f"{train_acc * 100:.2f}%", "type": model_type, "active": False})
    
    respModels.raise_for_status()
    data = respModels.json()
    model_id = data["data"]["ID"]

    if respModels.status_code == 200:
            st.success("Sukses membuat model!")
    else:
        st.error(resp.json().get("error", "Failed")) 
    
    for id in selected_ids:   
        respDatasetModels = requests.post(DATASET_MODEL_URL, json={"model_id": model_id, "dataset_id": id})
        
        if respDatasetModels.status_code == 200:
            st.success("Sukses membuat dataset - model!")
        else:
            st.error(resp.json().get("error", "Failed"))    