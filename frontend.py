import streamlit as st
import os
from llm.embeddings import mpnet_embeddings
from nlp_utils.nlp_engine import Preprocessor
from vectordb.pinecone_utils import PineconeUtils
from llm.openai_utils import Prompts
from decouple import config
import json
import uuid

# Initialize PineconeUtils and OpenAI Prompts
pc = PineconeUtils(config('PINECONE_KEY'), "us-west4-gcp")
llm = Prompts(config('OPENAI_KEY'))

# Create the Streamlit app and set the title
st.title("Nirvana Labs - Rishi")

# Create tabs for upload and find functionalities
upload, find = st.tabs(["Upload ☁️", "Find 🔎"])

# upload tab
with upload:
    # Upload resume files
    uploaded_files = st.file_uploader("Choose Resume", accept_multiple_files=True, type=['pdf'])
    
    # Check if the "Upload" button is clicked
    if st.button("Upload"):
        with st.spinner():
            # Loop through the uploaded resume files
            for uploaded_file in uploaded_files:
                # Create the file path
                file_path = os.path.join("resume_dir", str(uploaded_file.id))

                # Save the file in the folder
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())

                # Extract text from the resume PDF
                resume_text = Preprocessor.extract_text_from_pdf(file_path)

                # Get the prompt for resume screening from OpenAI
                prompt = llm.resume_screening

                # Perform GPT-3 completion on the resume text concatenated with the prompt
                resume_screened = llm.gpt3_completion(resume_text + "\n" + prompt)

                # Convert the GPT-3 response to a dictionary
                try:
                    resume_dict = json.loads(resume_screened)

                    # Prepare embeddings and Pinecone vector objects for each resume section
                    vec_obj_lst = []
                    for resume_sec in resume_dict.values():
                        resume_clean_lst = Preprocessor.preprocess_resume(resume_sec)
                        resume_clean = ' '.join(resume_clean_lst)
                        embedding = mpnet_embeddings(resume_clean)
                        vec_obj = pc.create_vector_object(str(uuid.uuid4()), embedding, metadata={"file_id": uploaded_file.id})
                        vec_obj_lst.append(vec_obj)

                    # Upsert the vector objects to the Pinecone index
                    vec_upsert = pc.upsert_vectors(vec_obj_lst, "ats-gpt")
                    print(vec_upsert)
                except:
                    pass

# find tab
with find:
    # Text area for entering the job description
    job_desc = st.text_area('Enter Job Description')

    # Check if the "Top Applicants" button is clicked
    if st.button("Top Applicants 🔎"):
        # Get the prompt for job description search from OpenAI
        prompt = llm.jd_search

        # Perform GPT-3 completion on the job description text concatenated with the prompt
        job_des_analysis = llm.gpt3_completion(job_desc + prompt)

        # Preprocess the GPT-3 response for the job description
        job_desc_clean_lst = Preprocessor.preprocess_resume(job_des_analysis)
        job_desc_clean = ' '.join(job_desc_clean_lst)

        # Get embeddings for the preprocessed job description using MPNet
        jd_vecs = mpnet_embeddings(job_desc_clean)
        jd_vecs_lst = jd_vecs.tolist()

        # Search the Pinecone index for the top 10 matches to the job description embeddings
        search = pc.search_index("ats-gpt", 50, vector=jd_vecs_lst)

        # Find the most frequent file IDs in the search results
        file_id_lst = Preprocessor.find_frequent_file_id(search, 3)

        # Display the top candidates' resume text
        for file_id in file_id_lst:
            # Create the file path
            file_path = os.path.join("resume_dir", str(int(file_id)))

            # Use a Streamlit expander to show the content of each resume
            with st.expander("Applicant ID: " + str(file_id)):
                st.write(Preprocessor.extract_text_from_pdf(file_path))
