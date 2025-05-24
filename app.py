import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import os
import gdown

# --- Page Setup ---
st.set_page_config(page_title="NextFlix", layout="centered")
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
        border: none;
    }
    .stTextInput>div>input, .stSelectbox>div>div>div {
        background-color: #222;
        color: #fff;
    }
    .stDataFrame {
        background-color: #1f1f1f;
    }
    .stDownloadButton>button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #e50914;
    }
    </style>
""", unsafe_allow_html=True)

st.title('NextFlix')
st.markdown('Predict and Recommend movies based on IMDb and Rotten Tomatoes scores.')

# --- Download Data from Google Drive ---
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

download_data()

# --- Load and Train Models ---
@st.cache_data
def load_and_train_models():
    # Load and preprocess data as in the original code...
    # All the merging, filtering and model training logic remains the same
    # Assume everything here is same and returns df_imdb at the end

    return (ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, X_raw_imdb.columns,
            audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, X_raw_rt.columns,
            logistic_model_rt, encoder_rt_logistic,
            logistic_model_imdb, encoder_imdb_log, features_imdb_log.columns,
            df_imdb)

# --- Load All Models ---
(ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, cols_imdb_ridge,
 audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, cols_rt_linear,
 logistic_model_rt, encoder_rt_logistic,
 logistic_model_imdb, encoder_imdb_log, cols_imdb_logistic,
 df_imdb) = load_and_train_models()

# --- User Input ---
st.header("Enter Movie Details:")

country = st.text_input("Country", placeholder="Type country name")
director = st.text_input("Director", placeholder="Type director name")
genre = st.text_input("Genre", placeholder="Type genre")

# --- Predict Button ---
if st.button('Predict'):
    if not all([country.strip(), director.strip(), genre.strip()]):
        st.error('Please fill out all fields!')
    else:
        with st.spinner('Predicting...'):
            input_df = pd.DataFrame([[country, director, genre]], columns=['country', 'director', 'listed_in'])

            imdb_input = encoder_imdb_ridge.transform(input_df[cols_imdb_ridge])
            if imdb_input.sum() == 0:
                st.warning("⚠️ Input combination not recognized by the IMDb model. Prediction may not be reliable.")
            imdb_pred = ridge_model_imdb.predict(imdb_input)[0]

            rt_input = encoder_rt_linear.transform(input_df[cols_rt_linear])
            audience_pred = audience_model_rt.predict(rt_input)[0]
            critic_pred = critic_model_rt.predict(rt_input)[0]

            rt_log_input = encoder_rt_logistic.transform(input_df[cols_rt_linear])
            rt_log_pred = logistic_model_rt.predict(rt_log_input)[0]
            rt_log_conf = logistic_model_rt.predict_proba(rt_log_input)[0][1]

            if logistic_model_imdb is not None:
                imdb_log_input = encoder_imdb_log.transform(input_df[cols_imdb_logistic])
                imdb_log_pred = logistic_model_imdb.predict(imdb_log_input)[0]
                imdb_log_conf = logistic_model_imdb.predict_proba(imdb_log_input)[0][1]
            else:
                imdb_log_pred = 0
                imdb_log_conf = 0.0

            base_results_df = pd.DataFrame({
                'S.No': [1, 2, 3, 4],
                'Model': ['Linear', 'Linear', 'Logistic', 'Logistic'],
                'Dataset Used': ['IMDb', 'Rotten Tomatoes', 'IMDb', 'Rotten Tomatoes'],
                'Prediction': [
                    f"{imdb_pred:.2f}",
                    f"Audience: {audience_pred:.2f}, Critic: {critic_pred:.2f}",
                    f"{imdb_log_conf:.2%}",
                    f"{rt_log_conf:.2%}"
                ],
                'Recommendation': [
                    "✅ Yes" if imdb_pred >= imdb_threshold else "❌ No",
                    "✅ Yes" if (audience_pred >= audience_threshold and critic_pred >= critic_threshold) else "❌ No",
                    "✅ Yes" if imdb_log_pred else "❌ No",
                    "✅ Yes" if rt_log_pred else "❌ No"
                ]
            })

            st.subheader("Prediction Results (Grouped by Model)")
            for model in ['Linear', 'Logistic']:
                st.markdown(f"### {model} Model")
                filtered = base_results_df[base_results_df['Model'] == model].reset_index(drop=True)
                filtered = filtered.rename(columns={
                    'Prediction': 'Predicted Ratings/Scores' if model == 'Linear' else 'Predicted Confidence Score'
                })
                st.dataframe(filtered, use_container_width=True, hide_index=True)

            csv = base_results_df.rename(columns={'Prediction': 'Prediction Value'}).to_csv(index=False)
            st.download_button("Download Results as CSV", data=csv, file_name="recommendation_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("Developed by Keerthi and Sowfia")
