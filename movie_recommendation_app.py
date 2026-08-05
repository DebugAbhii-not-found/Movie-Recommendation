import os
import pickle

import pandas as pd
import streamlit as st

VECTORIZER_PATH = "movie_vectorizer.pkl"
SIMILARITY_PATH = "similarity_matrix.pkl"
MOVIES_PATH = "movies_data.pkl"


@st.cache_resource
def load_artifacts():
    paths = [VECTORIZER_PATH, SIMILARITY_PATH, MOVIES_PATH]
    if not all(os.path.exists(p) for p in paths):
        return None, None
    with open(MOVIES_PATH, "rb") as f:
        movies = pickle.load(f)
    with open(SIMILARITY_PATH, "rb") as f:
        similarity_matrix = pickle.load(f)
    return movies, similarity_matrix


def recommend_movies(movie_title, movies, similarity_matrix, number_of_recommendations=5):
    if movie_title not in movies["title"].values:
        return pd.DataFrame()

    movie_index = movies.index[movies["title"] == movie_title][0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = [item for item in similarity_scores if item[0] != movie_index]

    recommendations = []
    for index, score in similarity_scores[:number_of_recommendations]:
        recommendations.append(
            {
                "movie": movies.iloc[index]["title"],
                "similarity": round(score, 3),
            }
        )
    return pd.DataFrame(recommendations)


st.set_page_config(page_title="Movie Recommendation", page_icon="🎬", layout="centered")
st.title("🎬 Content-Based Movie Recommendation System")
st.write(
    "Pick a movie you like, and the system will suggest similar titles based "
    "on shared genres and themes in their descriptions."
)

movies, similarity_matrix = load_artifacts()

if movies is None or similarity_matrix is None:
    st.error(
        f"Model files not found in this folder.\n\n"
        f"Expected: `{VECTORIZER_PATH}`, `{SIMILARITY_PATH}`, and `{MOVIES_PATH}`.\n\n"
        f"Run the **Save Model for Streamlit App** cell at the end of the "
        f"`Project4_Movie_Recommendation.ipynb` notebook, then copy those three "
        f".pkl files into this same folder as `app.py`."
    )
    st.stop()

selected_movie = st.selectbox("Choose a movie", sorted(movies["title"].tolist()))
num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if st.button("Recommend", type="primary"):
    results = recommend_movies(selected_movie, movies, similarity_matrix, num_recs)
    if results.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader(f"Because you liked '{selected_movie}':")
        for _, row in results.iterrows():
            st.write(f"**{row['movie']}** — similarity score: {row['similarity']}")

st.divider()
st.subheader("All movies in the dataset")
st.dataframe(movies, use_container_width=True)
