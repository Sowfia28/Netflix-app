import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import os
import gdown

# --- Netflix Dark Theme CSS ---
st.markdown("""
<style>
    .main {
        background-color: #141414;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #1f1f1f;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #e50914;
    }
    input, textarea, .stTextInput > div > div > input {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #444;
    }
    .stButton button, .stDownloadButton button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1.2em;
        transition: 0.3s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background-color: #b20710;
        color: #ffffff;
    }
    .stDataFrame, .stTable {
        background-color: #1c1c1c;
        color: #ffffff;
        border: 1px solid #2a2a2a;
    }
    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Page Setup ---
st.set_page_config(page_title="NextFlix", layout="centered")
st.title('🎬 NextFlix')
st.markdown('Predict and Recommend movies based on IMDb and Rotten Tomatoes scores.')

# --- Download Data ---
@st.cache_resource
def download_data():
    files = {
        "Project_3_data.csv": "1oOwysFn83UOPg46TcGnvisWFJvAlgAmA",
        "title.basics.tsv": "1oy7Q7HzhD5HsWvJhWkBvxWWtXF6AHbZp",
        "title.ratings.tsv": "1kQ0KfL0XfFkgmmDiTXbBXMFWDokQCmnD",
        "title.crew.tsv": "1QKdJxZVIg_KRx6Rd8Bi63bQk1Y48wM6q",
        "name.basics.tsv": "1iAsW1ZPYYQpxVYq2_2cCQWQn8SVGRr_C",
        "movie_info.csv": "13fvosTfqx-atwdHvOhfdE3d3ypPsAd2w"
    }
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)

# Call download
download_data()

# Placeholder for your model training function
@st.cache_data
def load_and_train_models():
    return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], pd.DataFrame()

# Dummy return to allow frontend dev and test
(ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, cols_imdb_ridge,
 audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, cols_rt_linear,
 logistic_model_rt, encoder_rt_logistic,
 logistic_model_imdb, encoder_imdb_log, cols_imdb_logistic,
 df_imdb) = load_and_train_models()

# --- User Input ---
st.header("🎥 Enter Movie Details:")

country = st.text_input('Country', placeholder='Type a country...')
director = st.text_input('Director', placeholder='Type a director name...')
genre = st.text_input('Genre', placeholder='Type a genre (e.g. Drama, Comedy)...')

# --- Predict Button ---
if st.button('🔍 Predict'):
    if not all([country.strip(), director.strip(), genre.strip()]):
        st.error('Please fill out all fields!')
    else:
        st.success("Prediction completed. This is a UI demo.")

        # Simulated output
        base_results_df = pd.DataFrame({
            'S.No': [1, 2, 3, 4],
            'Model': ['Linear', 'Linear', 'Logistic', 'Logistic'],
            'Dataset Used': ['IMDb', 'Rotten Tomatoes', 'IMDb', 'Rotten Tomatoes'],
            'Prediction': [
                "7.5",
                "Audience: 82.0, Critic: 79.0",
                "89.2%",
                "76.4%"
            ],
            'Recommendation': ["✅ Yes", "✅ Yes", "✅ Yes", "❌ No"]
        })

        st.subheader("📊 Prediction Results")
        for model in ['Linear', 'Logistic']:
            st.markdown(f"### {model} Model")
            filtered = base_results_df[base_results_df['Model'] == model].reset_index(drop=True)
            filtered = filtered.rename(columns={
                'Prediction': 'Predicted Ratings/Scores' if model == 'Linear' else 'Predicted Confidence Score'
            })
            st.dataframe(filtered, use_container_width=True, hide_index=True)

        csv = base_results_df.rename(columns={'Prediction': 'Prediction Value'}).to_csv(index=False)
        st.download_button("⬇ Download Results as CSV", data=csv, file_name="recommendation_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("<center>Developed by Keerthi and Sowfia</center>", unsafe_allow_html=True)
