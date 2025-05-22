import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import os
import requests
import gc  # Garbage collector

# --- Page Setup ---
st.set_page_config(page_title="NextFlix", layout="centered")
st.title('NextFlix')
st.markdown('Predict and Recommend movies based on IMDb and Rotten Tomatoes scores.')

# --- Download IMDb Datasets ---
@st.cache_resource
def download_data():
    imdb_urls = {
        "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
        "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
        "title.crew.tsv.gz": "https://datasets.imdbws.com/title.crew.tsv.gz",
        "name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz",
        "Project_3_data.csv": "https://drive.google.com/uc?id=1oOwysFn83UOPg46TcGnvisWFJvAlgAmA",
        "movie_info.csv": "https://drive.google.com/uc?id=13fvosTfqx-atwdHvOhfdE3d3ypPsAd2w"
    }

    for filename, url in imdb_urls.items():
        if not os.path.exists(filename):
            if "drive.google.com" in url:
                import gdown
                gdown.download(url, filename, quiet=False)
            else:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

download_data()

# --- Load and Train Models ---
@st.cache_data
def load_and_train_models():
    gc.enable()  # Ensure garbage collection is on

    # [Model training code remains unchanged and is assumed present here]

    return (
        ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, X_raw_imdb.columns,
        audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, X_raw_rt.columns,
        logistic_model_rt, encoder_rt_logistic,
        logistic_model_imdb, encoder_imdb_log, features_imdb_log.columns
    )

# --- Load All Models ---
(ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, cols_imdb_ridge,
 audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, cols_rt_linear,
 logistic_model_rt, encoder_rt_logistic,
 logistic_model_imdb, encoder_imdb_log, cols_imdb_logistic) = load_and_train_models()

# --- User Input ---
st.header("Enter Movie Details:")
country = st.text_input('Country', help="Example: United States")
director = st.text_input('Director', help="Example: Christopher Nolan")
genre = st.text_input('Genre', help="Example: Drama")

# --- Predict Button ---
if st.button('Predict'):
    if not all([country.strip(), director.strip(), genre.strip()]):
        st.error('Please fill out all fields!')
    else:
        with st.spinner('Predicting...'):
            input_df = pd.DataFrame([[country, director, genre]], columns=['country', 'director', 'listed_in'])

            imdb_input = encoder_imdb_ridge.transform(input_df[cols_imdb_ridge])
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
