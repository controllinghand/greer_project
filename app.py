# app.py
import streamlit as st
from app_nav import build_pages

pages, _PAGE_MAP = build_pages()
st.navigation(pages, position="sidebar").run()
