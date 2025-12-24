import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ff4b4b;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 30px;
    }
    .movie-box {
        background-color: #1f2933;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based content recommendation using Machine Learning</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# ---------------- ML MODEL ----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------------- FUNCTION ----------------
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# ---------------- UI ----------------
st.markdown("### 🎥 Select a movie")
movie_name = st.selectbox("", movies['title'].values)

if st.button("🚀 Recommend Movies"):
    with st.spinner("Finding best recommendations..."):
        recs = recommend(movie_name)

    if len(recs) == 0:
        st.error("Movie not found!")
    else:
        st.success("Top Recommended Movies")
        for movie in recs:
            st.markdown(f'<div class="movie-box">👉 {movie}</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>Made by Arthik Dwivedi</center>",
    unsafe_allow_html=True
)
