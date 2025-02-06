import streamlit as st
import PyPDF2
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords & stem words
    return " ".join(words)

# Function to calculate match score and extract keywords
def calculate_match_score(job_description, resume_texts):
    documents = [job_description] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Extracting top matching keywords
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(documents)
    feature_names = count_vectorizer.get_feature_names_out()
    
    matching_keywords = []
    for i, resume_text in enumerate(resume_texts):
        common_keywords = set()
        job_description_vector = count_matrix[0].toarray()[0]
        resume_vector = count_matrix[i + 1].toarray()[0]
        for j, word in enumerate(feature_names):
            if job_description_vector[j] > 0 and resume_vector[j] > 0:
                common_keywords.add(word)
        matching_keywords.append(common_keywords)
    
    return cosine_similarities, matching_keywords

# Streamlit app
def main():
    st.set_page_config(page_title="AI Resume Matcher 🚀", layout="wide")
    st.title("AI-Powered Resume Matcher 🚀")
    st.write("Upload a job description and resumes to get match scores and insights.")

    # Upload Job Description
    st.header("Step 1: Upload Job Description")
    job_description_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])

    # Upload Resumes
    st.header("Step 2: Upload Resumes")
    resume_files = st.file_uploader("Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

    if job_description_file and resume_files:
        # Extract and preprocess job description
        job_description = extract_text_from_pdf(job_description_file) if job_description_file.type == "application/pdf" else extract_text_from_docx(job_description_file)
        job_description = preprocess_text(job_description)

        # Extract and preprocess resumes
        resume_texts = []
        resume_names = []
        for resume_file in resume_files:
            resume_text = extract_text_from_pdf(resume_file) if resume_file.type == "application/pdf" else extract_text_from_docx(resume_file)
            resume_texts.append(preprocess_text(resume_text))
            resume_names.append(resume_file.name)

        # Calculate match scores and keywords
        match_scores, matching_keywords = calculate_match_score(job_description, resume_texts)

        # Display results
        st.header("Match Results")
        results = []
        for i, (score, keywords) in enumerate(zip(match_scores, matching_keywords)):
            st.subheader(f"📄 {resume_names[i]}")
            st.write(f"**Match Score:** {score:.2f}")
            st.write(f"**Matching Keywords:** {', '.join(keywords)}")
            st.progress(score)
            results.append({"Resume": resume_names[i], "Match Score": f"{score:.2f}", "Matching Keywords": ", ".join(keywords)})
        
        # Display Top Matching Resume
        best_match_index = match_scores.argmax()
        st.success(f"🏆 Best Matched Resume: {resume_names[best_match_index]} with Score: {match_scores[best_match_index]:.2f}")
        
        # Download results as CSV
        st.header("Download Results")
        results_df = pd.DataFrame(results)
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="Download Match Results as CSV", data=csv, file_name="resume_match_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
