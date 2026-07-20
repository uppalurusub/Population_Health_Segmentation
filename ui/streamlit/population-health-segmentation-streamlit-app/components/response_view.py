
import streamlit as st
def display_response(result):
    data=result.get("data",{})
    c1,c2=st.columns(2)
    c1.metric("Cluster",data.get("cluster","-"))
    c2.metric("Segment",data.get("segment_name","-"))
    with st.expander("Raw Response"):
        st.json(result)
