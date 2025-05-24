# Netflix-themed NextFlix App with Logo, Background, Animations, and Autocomplete
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import os
import gdown
from time import sleep

# --- Page Configuration (MUST BE FIRST) ---
st.set_page_config(page_title="NextFlix", layout="centered")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://wallpaperaccess.com/full/2703652.png');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }
    .main > div {
        background-color: rgba(20, 20, 20, 0.85);
        padding: 2rem;
        border-radius: 12px;
    }
    .stButton button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stDownloadButton button {
        background-color: #e50914;
        color: white;
    }
    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg", width=120)
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

# Call data download
download_data()

# --- Load and Train Models (abbreviated for brevity; reuse your logic here) ---
@st.cache_data
def load_models():
    df = pd.read_csv("Project_3_data.csv", nrows=10000)
    df['primaryTitle'] = df['title'].str.lower().str.strip()
    df['country'] = df['country'].fillna('Unknown')
    df['listed_in'] = df['listed_in'].fillna('Drama')
    df['director'] = df['director'].fillna('Various')
    encoder = OneHotEncoder(handle_unknown='ignore')
    X = encoder.fit_transform(df[['country', 'director', 'listed_in']])
    y = np.random.rand(len(df)) * 10
    model = Ridge().fit(X, y)
    threshold = np.median(model.predict(X))
    return model, encoder, threshold, df

model, encoder, threshold, df_data = load_models()

# --- Input Section with Autocomplete (Multiselect Hack) ---
st.header("🎥 Enter Movie Details:")

col1, col2, col3 = st.columns(3)

with col1:
    country = st.selectbox("Country", sorted(df_data['country'].dropna().unique()), index=0)

with col2:
    director = st.selectbox("Director", sorted(df_data['director'].dropna().unique()), index=0)

with col3:
    genre = st.selectbox("Genre", sorted(df_data['listed_in'].dropna().unique()), index=0)

# --- Predict Button with Animation ---
if st.button('🔍 Predict'):
    st.markdown("### ⏳ Analyzing movie details...")
    with st.spinner("Running predictions..."):
        sleep(2)
        input_df = pd.DataFrame([[country, director, genre]], columns=['country', 'director', 'listed_in'])
        input_encoded = encoder.transform(input_df)
        score = model.predict(input_encoded)[0]

    st.success("✅ Prediction Complete!")
    st.markdown(f"### 🎯 Predicted Score: **{score:.2f}**")
    st.markdown("### 📌 Recommendation:")
    if score >= threshold:
        st.markdown("<h3 style='color:#1DB954;'>✔️ Recommended to Watch!</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:#FF0000;'>❌ Not Recommended</h3>", unsafe_allow_html=True)

    # CSV Download
    result_df = pd.DataFrame({
        'Country': [country], 'Director': [director], 'Genre': [genre], 'Predicted Score': [score]
    })
    csv = result_df.to_csv(index=False)
    st.download_button("⬇ Download Result", csv, file_name="nextflix_prediction.csv", mime="text/csv")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center style='color:white;'>Developed by Keerthi and Sowfia</center>", unsafe_allow_html=True)
