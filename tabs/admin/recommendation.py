import math
import joblib
import streamlit as st
import os
import requests

API_URL = "http://localhost:8080/api/v1/recommendations"
DISEASE_URL = "http://localhost:8080/api/v1/disease"

def recommendation_tab():
    if "page" not in st.session_state:
        st.session_state.page = 1
        
    st.header("Manajamen Rekomendasi Kesahatan Paru-Paru")

    resp = requests.get(DISEASE_URL)

    diseases = resp.json()
    
    res = requests.get(API_URL)

    if res.status_code == 200:
        recommendations = res.json()
    else:
        st.error("Gagal mengambil data")
        st.stop()
        
    PAGE_SIZE = 5  
    total_data = len(recommendations)
    total_pages = math.ceil(total_data / PAGE_SIZE)

    start = (st.session_state.page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE

    page_data = recommendations[start:end]

    st.subheader("Daftar Recommendation")

    header = st.columns([1, 2, 2, 3, 2, 2])
    header[0].markdown("**ID**")
    header[1].markdown("**Disease**")
    header[2].markdown("**Type**")
    header[3].markdown("**Recommendation**")
    header[4].markdown("**Update**")
    header[5].markdown("**Delete**")

    for rec in page_data:
        cols = st.columns([1, 2, 2, 3, 2, 2])

        cols[0].write(rec["ID"])
        cols[1].write(rec["Disease"]["disease_name"])
        cols[2].write(rec["type"])
        cols[3].write(rec["recommendation_text"])

        # UPDATE
        if cols[4].button("✏️ Update", key=f"update_{rec['ID']}"):
            st.session_state["edit_id"] = rec["ID"]
            st.session_state["edit_data"] = rec    

        # DELETE
        if cols[5].button("🗑️ Delete", key=f"delete_{rec['ID']}"):
            del_res = requests.delete(f"{API_URL}/{rec['ID']}")
            if del_res.status_code == 200:
                st.success("Berhasil dihapus")
                st.rerun()
            else:
                st.error("Gagal menghapus")
                    
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Prev", disabled=st.session_state.page == 1):
            st.session_state.page -= 1
            st.rerun()

    with col2:
        st.markdown(
            f"<p style='text-align:center'>Page {st.session_state.page} of {total_pages}</p>",
            unsafe_allow_html=True
        )

    with col3:
        if st.button("Next ➡️", disabled=st.session_state.page == total_pages):
            st.session_state.page += 1
            st.rerun()
    

    if "edit_id" in st.session_state:
        st.subheader("Update Recommendation")

        data = st.session_state["edit_data"]

        new_type = st.selectbox(
            "Type",
            ["pencegahan", "pengobatan"],
            index=0 if data["type"] == "pencegahan" else 1
        )

        new_text = st.text_area(
            "Recommendation Text",
            value=data["recommendation_text"]
        )

        if st.button("💾 Simpan Perubahan"):
            payload = {
                "type": new_type,
                "recommendation_text": new_text,
                "disease_id": data["disease_id"]
            }

            upd_res = requests.put(
                f"{API_URL}/{data['id']}",
                json=payload
            )

            if upd_res.status_code == 200:
                st.success("Data berhasil diupdate")
                del st.session_state["edit_id"]
                del st.session_state["edit_data"]
                st.rerun()
            else:
                st.error("Gagal update data")

    disease_options = {d["disease_name"]: d["ID"] for d in diseases}
    
    type_options = {
        "Pencegahan": "pencegahan",
        "Pengobatan": "pengobatan"
    }

    st.subheader("Buat Recommendation Baru")

    with st.form("recommendation"):
        selected_name = st.selectbox("Pilih Penyakit", disease_options.keys())
        
        selected_type = st.selectbox("Pilih Tipe", type_options.keys())
        
        recommendation_text = st.text_input("Teks Rekomendasi")

        submit = st.form_submit_button("Submit")

    if submit:
        diseaseID = disease_options[selected_name]
        type = type_options[selected_type]
        resp = requests.post(API_URL, json={"disease_id": diseaseID, "type": type, "recommendation_text": recommendation_text})
        
        if resp.status_code == 200:
            st.success("Sukses membuat recommendasi!")
        else:
            st.error(resp.json().get("error", "Failed"))    
