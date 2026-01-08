import requests
import streamlit as st

HISTORY_URL = "http://localhost:8080/api/v1/history"

def history_tab():
    user_id = st.session_state.get("user_id")

    if not user_id:
        st.warning("Silakan login terlebih dahulu")
        st.stop()
    
    st.title("🔍 Riwayat Prediksi User")
    
    resp = requests.get(f"{HISTORY_URL}/{user_id}")

    histories = resp.json()
    
    # Header tabel
    header = st.columns([1, 2, 2, 2, 2, 2, 2])
    header[0].write("ID")
    header[1].write("Tanggal")
    header[2].write("Hasil Prediksi")
    header[3].write("Confidence")
    header[4].write("Penyakit")
    header[5].write("Gambar")
    header[6].write("Aksi")

    st.divider()

    for h in histories:
        cols = st.columns([1, 2, 2, 2, 2, 2, 2])

        cols[0].write(h["ID"])
        cols[1].write(h["CreatedAt"][:19])
        cols[2].write(h["prediction_result"])
        cols[3].write(h["confidence_result"])
        cols[4].write(h["Disease"]["disease_name"])

        # tampilkan gambar
        image_url = f"{h['image_path']}"
        cols[5].image(image_url, width=80)

        # === ACTION BUTTON ===
        with cols[6]:
            if st.button("🗑️ Delete", key=f"delete_{h['ID']}"):
                delete_resp = requests.delete(
                    f"{HISTORY_URL}/{h['ID']}"
                )

                if delete_resp.status_code == 200:
                    st.success("Data berhasil dihapus")
                    st.rerun()
                else:
                    st.error("Gagal menghapus data")
                    