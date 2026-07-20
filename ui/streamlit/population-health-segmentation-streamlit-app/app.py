
import streamlit as st
from components.sidebar import render_sidebar
from pages.home import render

st.set_page_config(page_title="Population Health Segmentation",layout="wide")
render_sidebar()
render()
