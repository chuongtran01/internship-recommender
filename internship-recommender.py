from bs4 import BeautifulSoup
import pandas as pd
from PyPDF2 import PdfReader
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st

ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'

POS_LIST = [NOUN, VERB, ADJ, ADV]


def main():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

    create_introduction()

    uploaded_file = st.file_uploader("Choose your .pdf file", type="pdf")

    if (uploaded_file is not None):
        resume = read_pdf(uploaded_file)

        job_data = return_data_list("./data/data_without_newline.csv")
        resume_text = remove_stop_words(resume)
        resume_text = lemmatize_text(resume_text)

        job_df = pre_process_data_job(job_data)
        resume_df = pre_process_resume(resume_text)

        top_10_recommended_job = return_top_recommended_jobs(
            10, resume_df, job_df)

        top_10_recommended_job_excluded = top_10_recommended_job.drop(
            'Job Description', axis=1)

        top_10_recommended_job_excluded = top_10_recommended_job_excluded.iloc[:, 1:]

        create_table(top_10_recommended_job_excluded)


def return_top_recommended_jobs(n, resume_text, job_df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['Job Description'])

    recommended_job = recommend_job(
        resume_text, tfidf_matrix, tfidf_vectorizer, job_df)

    return recommended_job[:n]


def return_data_list(csv_file):
    csvFile = pd.read_csv(csv_file)
    return csvFile


def create_table(data):
    st.table(data)


def create_introduction():
    url = "https://github.com/chuongtran01/internship-recommender"
    st.title("Hi! I am Chuong Tran")
    st.subheader("Internship Recommender Based on Resumes")
    st.markdown("[Github repository](%s)" % url)

    st.write("This recommendation system leverages Natural Language Processing techniques, including TF-IDF and Cosine Similarity, to match internship opportunities with user resume")


def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    page_one = reader.pages[0]
    resume = page_one.extract_text()
    return resume


def remove_stop_words(text):
    words_token = word_tokenize(str(text))
    stop_words = set(stopwords.words('english'))
    stop_words.add("’")
    filtered_words = [
        word for word in words_token if word.lower() not in stop_words]

    # Join the filtered words back into a text
    filtered_text = ' '.join(filtered_words)

    return filtered_text


def nltk_pos_tagger(nltk_tag):
    """
    Map NLTK POS tags to WordNet POS tags.

    Parameters:
    - nltk_tag (str): The POS tag assigned by NLTK.

    Returns:
    int or None: WordNet POS tag corresponding to the input NLTK POS tag, or None if not recognized.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()

    # Part-of-speech tagging using NLTK
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))

    # Map NLTK POS tags to WordNet POS tags
    # x is from nltk_tagged x[0] = CSS, x[1] = NNP => pass x[1] to nltk_pos_tagger fn
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)

    lemmatized_sentence = []

    # Lemmatize each word based on its POS tag
    for word, tag in wordnet_tagged:
        if tag is not None:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    # Join the lemmatized words back into a sentence
    return " ".join(lemmatized_sentence)


def keep_alpha_char(text):
    """
    Remove non-alphabetic characters from the input text, keeping only alphabetical characters.

    Parameters:
    - text (str): The input text containing alphanumeric and non-alphanumeric characters.

    Returns:
    str: A cleaned string containing only alphabetical characters.
    """
    alpha_only_string = re.sub(r'[^a-zA-Z]', ' ', text)
    cleaned_string = re.sub(r'\s+', ' ', alpha_only_string)
    return cleaned_string


def pre_process_resume(resume_text):
    """
    Preprocesses a resume text by removing stop words, lemmatizing, and keeping only alphabet characters.

    Parameters:
    - resume_text (str): The text content of the resume.

    Returns:
    str: The preprocessed resume text.
    """
    # Remove non-alphabetic characters and keep only alphabet characters
    resume_text = keep_alpha_char(resume_text)

    # Lemmatize the words in the resume text
    resume_text = lemmatize_text(resume_text)

    # Remove stop words from the resume text
    resume_text = remove_stop_words(resume_text)

    # Convert the resume text to lowercase
    resume_text = resume_text.lower()

    return resume_text


def recommend_job(resume_text, tfidf_matrix, tfidf_vectorizer, df):
    """
    Recommends jobs based on an input word using TF-IDF and cosine similarity.

    Parameters:
    - resume_text (str): The input word or text for which job recommendations are sought.
    - tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing job descriptions.
    - tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for vectorizing input words.
    - df (pd.DataFrame): The DataFrame containing job information.

    Returns:
    pd.DataFrame: A table of recommended jobs sorted by similarity to the input word.
    """
    # Calculate the TF-IDF vector for the input word
    resume_text_vector = tfidf_vectorizer.transform([resume_text])

    # Calculate cosine similarities between the input word vector and job vectors
    cosine_similarities = cosine_similarity(resume_text_vector, tfidf_matrix)

    # Get indices of jobs sorted by similarity (highest to lowest)
    job_indices = cosine_similarities.argsort()[0][::-1]

    # Extract the jobs corresponding to the top recommendations
    top_recommendations_full = [df.iloc[index] for index in job_indices]

    return pd.DataFrame(top_recommendations_full)


def pre_process_data_job(job_df):
    """
    Preprocess the job_df database by removing stop words, returning the words to their base form,
    and keeping only alphabet characters.

    Parameters:
    - job_df (pd.DataFrame): The job database containing job descriptions.
      job_df["Role"] is the column that contains the title and the role of the internship.

    Returns:
    pd.DataFrame: A table of recommended jobs that has the pre-processed data for the column "Role".
    """
    # Drop rows with missing values in the "Role" column
    job_df.dropna(subset=['Job Description'], inplace=True)

    # Apply preprocessing steps directly to the original DataFrame
    job_df['Job Description'] = job_df['Job Description'].apply(
        keep_alpha_char)
    job_df['Job Description'] = job_df['Job Description'].apply(lemmatize_text)
    job_df['Job Description'] = job_df['Job Description'].apply(
        remove_stop_words)
    job_df['Job Description'] = job_df['Job Description'].str.lower()

    # Removing string that has 🔒 (which means no longer active) in application/link
    # job_df["Application/Link"].replace('🔒', pd.NA, inplace=True)
    # job_df.dropna(inplace = True)

    return job_df


if __name__ == "__main__":
    main()
