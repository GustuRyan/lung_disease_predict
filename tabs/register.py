import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8080/api/v1/register"  

def register_tab():
    st.subheader("Register New Account")

    # Register Form 
    with st.form("register_form"):
        name = st.text_input("Name", placeholder="Enter your name")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        submit_btn = st.form_submit_button("Register")

    if submit_btn:
        if password != confirm_password:
            st.error("❌ Password and Confirm Password do not match!")
            return
        
        payload = {
            "name": name,
            "email": email,
            "password": password
        }

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 201 or response.status_code == 200:
                st.success("✅ Registration successful!")
                st.json(response.json())
            else:
                st.error("❌ Failed to register.")
                st.write("Response:", response.text)

        except Exception as e:
            st.error("❌ Error connecting to API.")
            st.write(e)