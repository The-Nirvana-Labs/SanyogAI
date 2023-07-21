import streamlit as st
import os
from llm.embeddings import mpnet_embeddings
from nlp_utils.nlp_engine import Preprocessor
from vectordb.pinecone_utils import PineconeUtils
from llm.openai_utils import Prompts
from decouple import config
import json
import uuid

pc = PineconeUtils(config('PINECONE_KEY'),"us-west4-gcp")
llm = Prompts(config('OPENAI_KEY'))

st.title("Resume Analysis")


upload, find = st.tabs(["Upload ☁️", "Find 🔎"])

# upload tab
with upload:
    # upload resume files
    uploaded_files = st.file_uploader("Choose Resume", accept_multiple_files=True, type=['pdf'])
    
    if st.button("Upload"):
        # loop through the resume files
        for uploaded_file in uploaded_files:
            # create file path
            file_path = os.path.join("resume_dir", str(uploaded_file.id))

            # save file in the folder
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            resume_text = Preprocessor.extract_text_from_pdf(file_path)
            prompt = llm.resume_screening 
            resume_screened = llm.gpt3_completion(resume_text +"\n"+ prompt)
            resume_dict = json.loads(resume_screened)
            vec_obj_lst = []
            for resume_sec in resume_dict.values():
                resume_clean_lst = Preprocessor.preprocess_resume(resume_sec)
                resume_clean = ' '.join(resume_clean_lst)
                embedding = mpnet_embeddings(resume_clean)
                vec_obj = pc.create_vector_object(str(uuid.uuid4()),embedding,metadata={"file_id":uploaded_file.id})
                vec_obj_lst.append(vec_obj)
            
            vec_upsert = pc.upsert_vectors(vec_obj_lst,"ats-gpt")
            print(vec_upsert)

            
        
           
       
with find:
    job_desc = st.text_area('Enter Job Description')
    if st.button("Top Applicants 🔎"):
        prompt = llm.jd_search
        job_des_analysis = llm.gpt3_completion(job_desc + prompt)
        job_desc_clean_lst = Preprocessor.preprocess_resume(job_des_analysis)
        job_desc_clean = ' '.join(job_desc_clean_lst)
        jd_vecs = mpnet_embeddings(job_desc_clean)
        jd_vecs_lst = jd_vecs.tolist()

        search = pc.search_index("ats-gpt",10,vector=jd_vecs_lst)
        
        file_id_lst = Preprocessor.find_frequent_file_id(search,3)
       
        for file_id in file_id_lst:
            # create file path

            file_path = os.path.join("resume_dir", str(int(file_id)))

            with st.expander(str(file_id)):
                st.write(Preprocessor.extract_text_from_pdf(file_path))
        
        
