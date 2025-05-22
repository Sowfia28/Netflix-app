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
    project_df = pd.read_csv("Project_3_data.csv", nrows=10000)
    project_df.rename(columns={'title': 'primaryTitle'}, inplace=True)
    project_df['primaryTitle'] = project_df['primaryTitle'].str.strip().str.lower()

    title_basics = pd.read_csv("title.basics.tsv", sep="\t", na_values="\\N", low_memory=False, nrows=10000)
    title_basics['primaryTitle'] = title_basics['primaryTitle'].str.strip().str.lower()

    # Debug output to verify title alignment
    st.markdown("### 🔎 Debug Preview of Titles")
    st.write("Project DataFrame Titles:", project_df['primaryTitle'].dropna().unique()[:5])
    st.write("Title Basics Titles:", title_basics['primaryTitle'].dropna().unique()[:5])

    title_ratings = pd.read_csv("title.ratings.tsv", sep="\t", na_values="\\N", low_memory=False, nrows=10000)
    title_crew = pd.read_csv("title.crew.tsv", sep="\t", na_values="\\N", nrows=10000)
    name_basics = pd.read_csv("name.basics.tsv", sep="\t", na_values="\\N", nrows=10000)
    df_info = pd.read_csv("movie_info.csv", nrows=10000)

    df_imdb = pd.merge(project_df, title_basics[['tconst', 'primaryTitle']], on='primaryTitle', how='left')
    if df_imdb['tconst'].isnull().all():
        st.error("❌ No matches found between project_df and title_basics on 'primaryTitle'. Check formatting.")
        st.stop()

    df_imdb = pd.merge(df_imdb, title_ratings[['tconst', 'averageRating']], on='tconst', how='left')
    df_imdb = pd.merge(df_imdb, title_crew[['tconst', 'directors']], on='tconst', how='left')
    df_imdb['director_id'] = df_imdb['directors'].str.split(',').str[0]
    df_imdb = pd.merge(df_imdb, name_basics[['nconst', 'primaryName']], left_on='director_id', right_on='nconst', how='left')
    df_imdb.rename(columns={'primaryName': 'director'}, inplace=True)
    df_imdb.dropna(subset=['country', 'listed_in', 'averageRating', 'director'], inplace=True)
    df_imdb = df_imdb.drop_duplicates(subset=['primaryTitle'])

    if df_imdb.empty:
        st.error("❌ df_imdb is empty after merging all IMDb data. Check input formats or dataset integrity.")
        st.stop()

    X_raw_imdb = df_imdb[['country', 'director', 'listed_in']]
    y_imdb = df_imdb['averageRating']
    encoder_imdb_ridge = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded_imdb = encoder_imdb_ridge.fit_transform(X_raw_imdb)
    ridge_model_imdb = Ridge(alpha=1.0).fit(X_encoded_imdb, y_imdb)
    imdb_threshold = np.median(ridge_model_imdb.predict(X_encoded_imdb))

    df_info['audience_score'] = df_info['audience_score'].str.rstrip('%').astype(float)
    df_info['critic_score'] = df_info['critic_score'].str.rstrip('%').astype(float)
    df_info['title'] = df_info['title'].str.strip().str.lower()
    project_df['title_lower'] = project_df['primaryTitle'].str.strip().str.lower()

    combined_rt = pd.merge(project_df, df_info[['title', 'audience_score', 'critic_score']], 
                           left_on='title_lower', right_on='title', how='inner')
    combined_rt.dropna(subset=['country', 'director', 'listed_in', 'audience_score', 'critic_score'], inplace=True)
    combined_rt = combined_rt.drop_duplicates(subset=['title'])

    X_raw_rt = combined_rt[['country', 'director', 'listed_in']]
    y_rt = combined_rt[['audience_score', 'critic_score']]
    encoder_rt_linear = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded_rt = encoder_rt_linear.fit_transform(X_raw_rt)
    audience_model_rt = LinearRegression().fit(X_encoded_rt, y_rt['audience_score'])
    critic_model_rt = LinearRegression().fit(X_encoded_rt, y_rt['critic_score'])
    audience_threshold = np.median(audience_model_rt.predict(X_encoded_rt))
    critic_threshold = np.median(critic_model_rt.predict(X_encoded_rt))

    df_logistic_rt = combined_rt.copy()
    df_logistic_rt['recommend'] = np.where(
        (df_logistic_rt['audience_score'] >= audience_threshold) &
        (df_logistic_rt['critic_score'] >= critic_threshold), 1, 0)
    X_log_rt = df_logistic_rt[['country', 'director', 'listed_in']]
    y_log_rt = df_logistic_rt['recommend']
    encoder_rt_logistic = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded_log_rt = encoder_rt_logistic.fit_transform(X_log_rt)
    X_train_rt, _, y_train_rt, _ = train_test_split(X_encoded_log_rt, y_log_rt, test_size=0.2, random_state=42)
    logistic_model_rt = LogisticRegression(max_iter=1000).fit(X_train_rt, y_train_rt)

    basics = title_basics.astype(str)
    ratings = title_ratings.astype(str)
    crew = title_crew.astype(str)
    names = name_basics.astype(str)
    project3 = project_df[['primaryTitle', 'country']].copy()

    movies = basics[basics['titleType'] == 'movie'].copy()
    movies['primaryTitle'] = movies['primaryTitle'].str.strip().str.lower()
    df_imdb_log = movies.merge(ratings, on='tconst', how='inner')
    df_imdb_log = df_imdb_log.merge(crew[['tconst', 'directors']], on='tconst', how='left')
    df_imdb_log.dropna(subset=['averageRating', 'genres', 'directors'], inplace=True)
    df_imdb_log['averageRating'] = df_imdb_log['averageRating'].astype(float)
    df_imdb_log['recommend'] = (df_imdb_log['averageRating'] >= df_imdb_log['averageRating'].median()).astype(int)
    df_imdb_log['directors'] = df_imdb_log['directors'].apply(lambda x: x.split(',')[0])
    id_to_name = dict(zip(names['nconst'], names['primaryName']))
    df_imdb_log['director_name'] = df_imdb_log['directors'].map(id_to_name)
    df_imdb_log = df_imdb_log.merge(project3, on='primaryTitle', how='left')
    df_imdb_log.dropna(subset=['country', 'director_name'], inplace=True)

    features_imdb_log = df_imdb_log[['country', 'director_name', 'genres']]
    features_imdb_log.columns = ['country', 'director', 'listed_in']
    target_imdb_log = df_imdb_log['recommend']
    encoder_imdb_log = OneHotEncoder(handle_unknown='ignore')
    X_encoded_imdb_log = encoder_imdb_log.fit_transform(features_imdb_log)
    X_train_imdb, _, y_train_imdb, _ = train_test_split(X_encoded_imdb_log, target_imdb_log, test_size=0.2, random_state=42)
    logistic_model_imdb = LogisticRegression(max_iter=1000).fit(X_train_imdb, y_train_imdb)

    return (ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, X_raw_imdb.columns,
            audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, X_raw_rt.columns,
            logistic_model_rt, encoder_rt_logistic,
            logistic_model_imdb, encoder_imdb_log, features_imdb_log.columns)

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

            imdb_log_input = encoder_imdb_log.transform(input_df[cols_imdb_logistic])
            imdb_log_pred = logistic_model_imdb.predict(imdb_log_input)[0]
            imdb_log_conf = logistic_model_imdb.predict_proba(imdb_log_input)[0][1]

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
