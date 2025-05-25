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

# Load and Train Models
@st.cache_data
def load_and_train_models():
    download_data()
    project_df = pd.read_csv("Project_3_data.csv", nrows=50000)
    project_df.rename(columns={'title': 'primaryTitle'}, inplace=True)
    project_df['primaryTitle'] = project_df['primaryTitle'].str.strip().str.lower()

    title_basics = pd.read_csv("title.basics.tsv", sep="\t", na_values="\\N", low_memory=False, nrows=50000)
    title_basics['primaryTitle'] = title_basics['primaryTitle'].str.strip().str.lower()

    title_ratings = pd.read_csv("title.ratings.tsv", sep="\t", na_values="\\N", low_memory=False, nrows=50000)
    title_crew = pd.read_csv("title.crew.tsv", sep="\t", na_values="\\N", nrows=50000)
    name_basics = pd.read_csv("name.basics.tsv", sep="\t", na_values="\\N", nrows=50000)
    df_info = pd.read_csv("movie_info.csv", nrows=50000)

    df_imdb = pd.merge(project_df, title_basics[['tconst', 'primaryTitle']], on='primaryTitle', how='left')
    df_imdb = pd.merge(df_imdb, title_ratings[['tconst', 'averageRating']], on='tconst', how='left')
    df_imdb = pd.merge(df_imdb, title_crew[['tconst', 'directors']], on='tconst', how='left')
    df_imdb['director_id'] = df_imdb['directors'].str.split(',').str[0]
    df_imdb = pd.merge(df_imdb, name_basics[['nconst', 'primaryName']], left_on='director_id', right_on='nconst', how='left')
    df_imdb.rename(columns={'primaryName': 'director'}, inplace=True)
    df_imdb.dropna(subset=['country', 'listed_in', 'averageRating', 'director'], inplace=True)
    df_imdb = df_imdb.drop_duplicates(subset=['primaryTitle'])

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
    combined_rt = pd.merge(project_df, df_info[['title', 'audience_score', 'critic_score']],  left_on='title_lower', right_on='title', how='inner')
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
    df_logistic_rt['recommend'] = np.where((df_logistic_rt['audience_score'] >= audience_threshold) & (df_logistic_rt['critic_score'] >= critic_threshold), 1, 0)
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
    df_imdb_log = movies.merge(ratings, on='tconst', how='inner').merge(crew[['tconst', 'directors']], on='tconst', how='left')
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

    if len(target_imdb_log.unique()) < 2:
        logistic_model_imdb = None
        encoder_imdb_log = None
    else:
        encoder_imdb_log = OneHotEncoder(handle_unknown='ignore')
        X_encoded_imdb_log = encoder_imdb_log.fit_transform(features_imdb_log)
        X_train_imdb, _, y_train_imdb, _ = train_test_split(X_encoded_imdb_log, target_imdb_log, test_size=0.2, random_state=42)
        logistic_model_imdb = LogisticRegression(max_iter=1000).fit(X_train_imdb, y_train_imdb)

    return (ridge_model_imdb, encoder_imdb_ridge, imdb_threshold, X_raw_imdb.columns,
            audience_model_rt, critic_model_rt, encoder_rt_linear, audience_threshold, critic_threshold, X_raw_rt.columns,
            logistic_model_rt, encoder_rt_logistic,
            logistic_model_imdb, encoder_imdb_log, features_imdb_log.columns)

# --- Load All Models ---
models = load_and_train_models()

# --- User Input ---
st.header("Enter Movie Details:")
country = st.text_input('Country', help="Example: United States")
director = st.text_input('Director', help="Example: Christopher Nolan")
genre = st.text_input('Genre', help="Example: Drama")

if st.button('Predict'):
    if not all([country.strip(), director.strip(), genre.strip()]):
        st.error('Please fill out all fields!')
    else:
        with st.spinner('Predicting...'):
            input_df = pd.DataFrame([[country, director, genre]], columns=['country', 'director', 'listed_in'])
            try:
                imdb_input = models[1].transform(input_df[models[3]])
                imdb_pred = models[0].predict(imdb_input)[0]
            except Exception as e:
                imdb_pred = "N/A"
                st.warning(f"IMDb Linear Model error: {e}")
            try:
                rt_input = models[6].transform(input_df[models[9]])
                audience_pred = models[4].predict(rt_input)[0]
                critic_pred = models[5].predict(rt_input)[0]
            except Exception as e:
                audience_pred = critic_pred = "N/A"
                st.warning(f"RT Linear Model error: {e}")
            try:
                rt_log_input = models[11].transform(input_df[models[9]])
                rt_log_pred = models[10].predict(rt_log_input)[0]
                rt_log_conf = models[10].predict_proba(rt_log_input)[0][1]
            except Exception as e:
                rt_log_pred = 0
                rt_log_conf = 0.0
                st.warning(f"RT Logistic Model error: {e}")
            if models[12] is not None:
                try:
                    imdb_log_input = models[13].transform(input_df[models[14]])
                    imdb_log_pred = models[12].predict(imdb_log_input)[0]
                    imdb_log_conf = models[12].predict_proba(imdb_log_input)[0][1]
                except Exception as e:
                    imdb_log_pred = 0
                    imdb_log_conf = 0.0
                    st.warning(f"IMDb Logistic Model error: {e}")
            else:
                imdb_log_pred = 0
                imdb_log_conf = 0.0
            results = pd.DataFrame({
                'Model': ['Linear', 'Linear', 'Logistic', 'Logistic'],
                'Dataset': ['IMDb', 'RT', 'IMDb', 'RT'],
                'Prediction': [f"{imdb_pred}", f"Audience: {audience_pred}, Critic: {critic_pred}", f"{imdb_log_conf:.2%}", f"{rt_log_conf:.2%}"],
                'Recommendation': [
                    "✅ Yes" if imdb_pred != "N/A" and imdb_pred >= models[2] else "❌ No",
                    "✅ Yes" if audience_pred != "N/A" and critic_pred != "N/A" and audience_pred >= models[7] and critic_pred >= models[8] else "❌ No",
                    "✅ Yes" if imdb_log_pred else "❌ No",
                    "✅ Yes" if rt_log_pred else "❌ No"
                ]
            })
            st.dataframe(results)
            st.download_button("Download Results as CSV", data=results.to_csv(index=False), file_name="results.csv")

st.markdown("---")
st.markdown("Developed by Keerthi and Sowfia")
