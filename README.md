# Netflix Recommendation App - Dataset Documentation

This project uses a combination of Netflix, IMDb, and Rotten Tomatoes datasets to build machine learning models for:

- Genre classification
- Rating prediction
- Binary recommendation

---

## 1. Primary Dataset (Netflix Metadata)

**File**: `Project_3_data.csv`  
**Description**:
- Contains metadata for Netflix titles including:
  - `title`, `description`, `listed_in` (genre tags)
  - `director`, `country`, `release_year`
- Used for TF-IDF-based genre prediction and as input for regression/classification models.

---

## 2. IMDb Dataset Files (From IMDb Datasets)

| File Name           | Description |
|--------------------|-------------|
| `title.ratings.tsv`| IMDb user rating and vote count per title (used as regression target) |
| `title.basics.tsv` | Basic title info (type, year, runtime, genres) for feature merging |
| `title.crew.tsv`   | Directors and writers linked to each title |
| `name.basics.tsv`  | Name, profession, known-for titles of people (actors, directors, etc.) |

---

## 3. Rotten Tomatoes Dataset (From Kaggle)

**File**: `movie_info.csv`  
**Description**:
- Includes critic and audience scores
- Used to support rating prediction and recommendation model labeling

---

## 4. Machine Learning Tasks Enabled

- **Genre Prediction**: TF-IDF + Logistic Regression (Multi-label)
- **Rating Prediction**:
  - IMDb: Ridge Regression (R² = 0.87, RMSE = 0.44)
  - RT: Linear Regression
- **Recommendation Classification**:
  - IMDb model: F1 = 0.70, Recall = 0.73
  - RT model: Accuracy = 0.70, Recall = 0.48

---

## 5. Streamlit App

## Usage Notes

- Ensure all datasets are in the root or data directory
- Use Git LFS for files larger than 25MB
- Check the Streamlit app file for expected input formats
