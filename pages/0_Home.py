# pages/0_Home.py
import streamlit as st
from Home import render_home  # imports your big render_home() function

st.set_page_config(page_title="Greer Value Search", layout="wide")
render_home()
