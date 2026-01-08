import streamlit as st
import requests
from streamlit_cookies_manager import EncryptedCookieManager

API_URL = "http://localhost:8080/api/v1/login"

def login_tab(cookie_manager):
    st.subheader("Login")

    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        resp = requests.post(API_URL, json={"email": email, "password": password})

        if resp.status_code == 200:
            user = resp.json()["user"]

            cookie_manager["user_id"] = str(user["id"])
            cookie_manager["username"] = user["name"]
            cookie_manager["email"] = user["email"]
            cookie_manager["role"] = user.get("role", "user")
            cookie_manager.save()       

            st.success("Login success! Refresh halaman ✔")

        else:
            st.error(resp.json().get("error", "Failed"))
