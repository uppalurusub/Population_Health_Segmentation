
import streamlit as st
from components.request_form.form import render_form
from components.response_view import display_response
from api.patient_segmentation_api import predict
from utils.formatters import format_error

def render():
    st.title("🏥 Population Health Segmentation")
    submitted,payload=render_form()
    if submitted:
        try:
            res=predict(payload)
            display_response(res)
        except Exception as e:
            st.error(format_error(e))
