
import streamlit as st
def render_form():
    with st.form("predict_form"):
        payload={
            "age": st.number_input("Age",0,120,65),
            "gender": st.selectbox("Gender",["M","F"]),
            "bmi": st.number_input("BMI",10.0,80.0,28.0),
            "diabetes": st.selectbox("Diabetes",[0,1]),
            "hypertension": st.selectbox("Hypertension",[0,1]),
            "copd": st.selectbox("COPD",[0,1]),
            "mental_health": st.selectbox("Mental Health",[0,1]),
            "visits_per_year": st.number_input("Visits Per Year",0,100,5),
            "er_visits": st.number_input("ER Visits",0,50,0),
            "admissions": st.number_input("Admissions",0,50,0),
            "total_cost": st.number_input("Total Cost",0.0,1000000.0,5000.0)
        }
        submitted=st.form_submit_button("Predict Segment")
    return submitted,payload
