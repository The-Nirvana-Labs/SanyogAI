import streamlit as st



st.title("Resume Analysis")

upload, find = st.tabs(["Upload ☁️", "Find 🔎"])

with upload:
    uploaded_files = st.file_uploader("Choose Resume", accept_multiple_files=True, type=['pdf'])
    for uploaded_file in uploaded_files:
        st.write("filename:", uploaded_file.name)
       
with find:
    job_desc = st.text_area('Enter Job Description')
    st.button("Top Applicants 🔎")
