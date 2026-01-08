import math
import joblib
import requests
import streamlit as st
import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from model_cnn import train_dynamic_classes
from model_svm import train_svm_dynamic_classes

DATASET_URL = "http://localhost:8080/api/v1/dataset"
MODEL_URL = "http://localhost:8080/api/v1/model"
DATASET_MODEL_URL = "http://localhost:8080/api/v1/dataset-model"


def training_tab():

    # ==========================
    # LOAD DATASET FROM API
    # ==========================
    respo = requests.get(MODEL_URL)

    if respo.status_code != 200:
        st.error("Gagal mengambil data model")
        st.stop()

    df = pd.DataFrame(respo.json())

    st.title("📦 Manajemen Model")

    if df.empty:
        st.info("Belum ada model")
        st.stop()

    # ==========================
    # NORMALISASI ACTIVE
    # ==========================
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if x is None:
            return False
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes")
        if isinstance(x, (int, float)):
            return x == 1
        return False

    if "Active" in df.columns:
        df["active"] = df["Active"].apply(to_bool)
    elif "active" in df.columns:
        df["active"] = df["active"].apply(to_bool)
    else:
        df["active"] = False

    # ==========================
    # PAGINATION SETUP
    # ==========================
    PAGE_SIZE = st.selectbox("Jumlah data per halaman", [5, 10, 20], index=1)

    if "page" not in st.session_state:
        st.session_state.page = 1

    total_pages = math.ceil(len(df) / PAGE_SIZE)

    start = (st.session_state.page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    df_page = df.iloc[start:end]

    # ==========================
    # HEADER
    # ==========================
    header = st.columns([3, 4, 2, 2, 2, 3])
    header[0].markdown("**Model Name**")
    header[1].markdown("**File Path**")
    header[2].markdown("**Accuracy**")
    header[3].markdown("**Type**")
    header[4].markdown("**Active**")
    header[5].markdown("**Action**")

    st.divider()

    # ==========================
    # ISI TABEL
    # ==========================
    for _, row in df_page.iterrows():
        cols = st.columns([3, 4, 2, 2, 2, 3])

        cols[0].write(row["model_name"])
        cols[1].write(row["file_path"])
        cols[2].write(row["accuracy"])
        cols[3].write(row["type"])

        toggle_key = f"active_{row['ID']}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = row["active"]

        new_active = cols[4].toggle("", key=toggle_key)

        if new_active != row["active"]:
            update = requests.put(
                f"{MODEL_URL}/{row['ID']}",
                json={"active": new_active}
            )
            if update.status_code == 200:
                st.toast("Status model diperbarui", icon="✅")
                st.rerun()
            else:
                st.error("Gagal update status")

        # ===== ACTION BUTTON =====
        with cols[5]:
            c1, c2 = st.columns(2)

            if c2.button("🗑️ Delete", key=f"del_{row['ID']}"):
                confirm = st.warning(
                    f"Yakin hapus model **{row['model_name']}**?",
                    icon="⚠️"
                )
                if st.button("✅ Ya, Hapus", key=f"confirm_{row['ID']}"):
                    res = requests.delete(f"{MODEL_URL}/{row['ID']}")
                    if res.status_code == 200:
                        st.success("Model berhasil dihapus")
                        st.rerun()
                    else:
                        st.error("Gagal menghapus model")

    # ==========================
    # PAGINATION CONTROL
    # ==========================
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Prev") and st.session_state.page > 1:
            st.session_state.page -= 1
            st.rerun()

    with col2:
        st.markdown(
            f"<center>Halaman {st.session_state.page} / {total_pages}</center>",
            unsafe_allow_html=True
        )

    with col3:
        if st.button("Next ➡️") and st.session_state.page < total_pages:
            st.session_state.page += 1
            st.rerun()
    
    resp = requests.get(DATASET_URL)
    datasets = resp.json()
    dataset_options = {d["dataset_name"]: d["ID"] for d in datasets}

    st.header("🧠 Train Model dari Dataset Lokal (CNN / SVM)")

    dataset_root = "datasets"

    # ==========================
    # LOAD CLASSES
    # ==========================
    all_classes = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    selected_classes = st.multiselect(
        "Pilih Class yang akan digunakan untuk Training",
        all_classes,
        default=all_classes[:2],
    )

    selected_ids = [
        dataset_options[c] for c in selected_classes if c in dataset_options
    ]

    model_type = st.radio("Pilih Model", ["CNN", "SVM (HOG)"])

    model_name = st.text_input(
        "Nama file model (tanpa ekstensi)",
        "model_flexible"
    )

    epochs = st.slider("Jumlah Epoch", 1, 50, 10)
    batch_size = st.slider("Batch Size", 8, 64, 32)

    train_btn = st.button("🚀 Mulai Training")

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
            history, model, class_names, report, cm = train_dynamic_classes(
                selected_classes,
                dataset_root,
                epochs,
                batch_size,
                progress_callback=lambda e, t: progress.progress((e + 1) / t),
                status_callback=lambda msg: status_area.write(msg),
            )

            train_acc = history["train_acc"][-1]
            final_acc = history["val_acc"][-1]

            save_path = f"models/cnn/{model_name}.joblib"
            os.makedirs("models/cnn", exist_ok=True)

            joblib.dump(
                {
                    "model_type": "cnn",
                    "state_dict": model.state_dict(),
                    "classes": class_names,
                },
                save_path,
            )

        else:  # ================= SVM =================

            history, model, class_names, report, cm = train_svm_dynamic_classes(
                selected_classes,
                dataset_root,
                progress_callback=lambda p: progress.progress(p),
                status_callback=lambda msg: status_area.write(msg),
            )

            train_acc = history["train_acc"]
            final_acc = history["test_acc"]

            save_path = f"models/svm/{model_name}.joblib"
            os.makedirs("models/svm", exist_ok=True)

            joblib.dump(
                {
                    "model_type": "svm",
                    "model": model,
                    "classes": class_names
                },
                save_path,
            )


    # ==========================
    # RESULT
    # ==========================
    st.success("🎉 Training selesai!")

    st.subheader("📊 Hasil Evaluasi Model")
    st.write(f"Model Type: **{model_type}**")

    st.metric("Train Accuracy", f"{train_acc * 100:.2f}%")
    st.metric("Validation Accuracy", f"{final_acc * 100:.2f}%")

    st.write("💾 Model disimpan sebagai:")
    st.code(save_path)

    # ==========================
    # EVALUATION DETAIL
    # ==========================
    if report is not None and cm is not None:

        st.subheader("🔍 Precision • Recall • F1-Score")

        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.format("{:.4f}"))

        st.subheader("🧩 Confusion Matrix")

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)


    # ==========================
    # SAVE TO BACKEND
    # ==========================
    respModels = requests.post(
        MODEL_URL,
        json={
            "model_name": model_name,
            "file_path": save_path,
            "accuracy": f"{train_acc * 100:.2f}%",
            "type": model_type,
            "active": False,
        },
    )

    respModels.raise_for_status()
    model_id = respModels.json()["data"]["ID"]

    st.success("✅ Model berhasil disimpan ke backend")

    for did in selected_ids:
        resp = requests.post(
            DATASET_MODEL_URL,
            json={"model_id": model_id, "dataset_id": did},
        )

        if resp.status_code == 200:
            st.success(f"Dataset ID {did} terhubung ke model")
        else:
            st.error("Gagal menghubungkan dataset")
