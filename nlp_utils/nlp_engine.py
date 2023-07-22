import fitz 
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class Preprocessor:
        # Add more attributes here if needed
    @staticmethod
    def extract_text_from_pdf(path):
        text = ""
        with fitz.open(path) as pdf_document:
            num_pages = pdf_document.page_count

            for page_num in range(num_pages):
                page = pdf_document[page_num]
                text += page.get_text()

        return text
    
    @staticmethod
    def preprocess_resume(resume_text):
        '''
        Input:
            resume_text: a string containing resume text
        Output:
            resume_clean: a list of words containing the processed resume text
        '''
        stemmer = PorterStemmer()
        stopwords_english = set(stopwords.words('english'))


        resume_text = str(resume_text)
        # Convert the entire resume text to lowercase
        resume_text = resume_text.lower()

        # Remove email addresses (if any)
        resume_text = re.sub(r'\S+@\S+', '', resume_text)

        # Remove URLs
        resume_text = re.sub(r'http\S+|www\S+|https\S+', '', resume_text)

        # Remove digits (phone numbers and other numerical values)
        resume_text = re.sub(r'\d+', '', resume_text)

        # Remove punctuation (except for hyphens to handle words like "well-known")
        resume_text = resume_text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))

        # Tokenize the resume text
        resume_tokens = word_tokenize(resume_text)

        resume_clean = []
        for word in resume_tokens:
            if (word not in stopwords_english and
                len(word) > 1):  # Remove single-character words
                stem_word = stemmer.stem(word)  # Stemming word
                resume_clean.append(stem_word)
        
        return resume_clean
    
    @staticmethod
    def find_frequent_file_id(search_result, k):
        file_id_count = {}
        
        # Count the occurrences of each file_id in the search result
        for match in search_result['matches']:
            file_id = match['metadata']['file_id']
            if file_id in file_id_count:
                file_id_count[file_id] += 1
            else:
                file_id_count[file_id] = 1

        # Sort the file_id_count dictionary based on frequency in descending order
        sorted_file_ids = sorted(file_id_count.items(), key=lambda x: x[1], reverse=True)

        # Get the top k most frequent file_ids
        top_k_file_ids = [file_id for file_id, _ in sorted_file_ids[:k]]
    
        return top_k_file_ids
    
    def extract_n_preprocess(path):
        resume_text = Preprocessor.extract_text_from_pdf(path)
        resume_clean = Preprocessor.preprocess_resume(resume_text)

        return resume_clean


# text = Preprocessor.preprocess_resume("the candidate has worked as a market data analyst intern at sawo labs and as an operations data analyst intern at shopg they have experience in web scraping data analysis market trends consumer behavior and collaborating with teams")
# print(text)
    
