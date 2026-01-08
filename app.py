import streamlit as st
import pandas as pd
import joblib
from streamlit_cookies_manager import EncryptedCookieManager

from tabs.admin.dataset import dataset_tab
from tabs.admin.recommendation import recommendation_tab
from tabs.admin.training import training_tab
from tabs.login import login_tab
from tabs.register import register_tab
from tabs.user.dashboard import dashboard_tab
from tabs.user.history import history_tab

st.set_page_config(page_title="Lung Care System", layout="wide")

# =====================
#  COOKIE MANAGER
# =====================
cookie_manager = EncryptedCookieManager(
    prefix="lungcare_",
    password="your-strong-secret-password"
)

if not cookie_manager.ready():
    st.info("🔄 Menyiapkan sesi...")
    st.stop()

user_id = cookie_manager.get("user_id")

if user_id:
    st.session_state.user_id = int(user_id)
username = cookie_manager.get("username")
email = cookie_manager.get("email")
role = cookie_manager.get("role")

st.title("🔷Lung Disease Prediction🔷")
st.write(
    "Use this app to predict lung disease with a chest X-ray image, "
    "or need lung healthcare recommendations."
)

# =====================
#  SIDEBAR LOGIN STATUS
# =====================
if username:
    st.sidebar.success(f"Logged in as: {username}")
else:
    st.sidebar.warning("Not logged in.")

st.sidebar.title("📊 Lung Care System")
st.sidebar.markdown("---")

if "active_menu" not in st.session_state:
    st.session_state.active_menu = None

def set_active(value):
    st.session_state.active_menu = value


# ==========================
# AUTH SECTION
# ==========================
if not username:
    st.sidebar.subheader("Auth")

    auth_options = ["🔍 Login Account", "📈 Register Account"]

    auth_choice = st.sidebar.radio(
        "Authentication",
        auth_options,
        key="auth_radio",
        index=auth_options.index(st.session_state.active_menu)
            if st.session_state.active_menu in auth_options else 0,
        on_change=lambda: set_active(st.session_state.auth_radio)
    )

    if st.session_state.active_menu == "🔍 Login Account":
        login_tab(cookie_manager)

    elif st.session_state.active_menu == "📈 Register Account":
        register_tab()


# ==========================
# MAIN MENU SECTION
# ==========================
st.sidebar.subheader("Menu")

menu_options = ["🏠 Dashboard", "📂 History"]

menu_choice = st.sidebar.radio(
    "Main Menu",
    menu_options,
    key="main_menu_radio",
    index=menu_options.index(st.session_state.active_menu)
          if st.session_state.active_menu in menu_options else 0,
    on_change=lambda: set_active(st.session_state.main_menu_radio)
)

if st.session_state.active_menu == "🏠 Dashboard":
    dashboard_tab()

elif st.session_state.active_menu == "📂 History":
    history_tab()

# ==========================
# ADMIN MENU SECTION
# ==========================
if role == "admin":
    st.sidebar.subheader("Admin")

    admin_options = ["📊 Train Model", "📂 Manage Dataset", "📂 Manage Recommendation"]

    admin_choice = st.sidebar.radio(
        "Admin Menu",
        admin_options,
        key="admin_menu_radio",
        index=admin_options.index(st.session_state.active_menu)
            if st.session_state.active_menu in admin_options else 0,
        on_change=lambda: set_active(st.session_state.admin_menu_radio)
    )

    if st.session_state.active_menu == "📊 Train Model":
        training_tab()

    elif st.session_state.active_menu == "📂 Manage Dataset":
        dataset_tab()
        
    elif st.session_state.active_menu == "📂 Manage Recommendation":
        recommendation_tab()
        
# ============================
# LOGOUT SECTION (BOTTOM)
# ============================
st.sidebar.markdown("---")

if username:
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        cookie_manager["username"] = ""
        cookie_manager["email"] = ""
        cookie_manager["role"] = ""
        cookie_manager.save()
        st.success("You have been logged out.")
        st.rerun()
