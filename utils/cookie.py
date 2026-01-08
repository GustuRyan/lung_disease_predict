import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

@st.cache_resource
def _cookie_manager():
    return EncryptedCookieManager(
        prefix="lungcare_",
        password="your-strong-secret-password"
    )

def get_cookie_manager():
    cm = _cookie_manager()

    # JANGAN stop app
    if not cm.ready():
        st.info("🔄 Menyiapkan sesi...")
        st.stop()  # stop ringan, tapi ADA render info

    return cm

def get_user_id():
    cm = get_cookie_manager()
    user_id = cm.get("user_id")
    return int(user_id) if user_id else None
