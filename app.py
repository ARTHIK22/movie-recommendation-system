import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- API KEY ----------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_poster(movie_title):
    """Fetch movie poster from TMDB API"""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    return None


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.title {text-align:center;font-size:40px;font-weight:bold;color:#ff4b4b;}
.subtitle {text-align:center;font-size:18px;color:#cccccc;margin-bottom:30px;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Movie Recommendations with Posters & Genres</div>', unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")
movies = movies[['title', 'overview', 'genres']]
movies.dropna(inplace=True)

# Create combined data for better similarity accuracy
movies["combined"] = movies["overview"] + " " + movies["genres"]

# ---------------- ML MODEL ----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title, selected_genre, count):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:(count+1)]
    movie_indices = [i[0] for i in sim_scores]
    recs = movies.iloc[movie_indices]

    # Genre Filter (All = No Filter)
    if selected_genre != "All":
        recs = recs[recs['genres'].str.contains(selected_genre, case=False, na=False)]

    return recs


# ---------------- UI ----------------
st.markdown("### 🎭 Select Genre")
genres = ["All"] + sorted(movies['genres'].unique())
selected_genre = st.selectbox("Filter by Genre", genres)

st.markdown("### 🎥 Select a movie")
movie_name = st.selectbox("", movies['title'].values)

num_recs = st.slider("How many recommendations do you want?", 3, 10, 5)

if st.button("🚀 Recommend Movies"):
    with st.spinner("Finding the best movies for you..."):
        recs = recommend(movie_name, selected_genre, num_recs)

    if len(recs) == 0:
        st.error("No similar movies found 😕 Try another genre or movie.")
    else:
        st.success(f"🍿 Recommendations Based On **{movie_name}**:")
        cols = st.columns(3)
        for i, row in recs.iterrows():
            movie = row['title']
            poster = fetch_poster(movie)
            with cols[i % 3]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{movie}** 🎬")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Made by <b>Arthik Dwivedi</b></center>", unsafe_allow_html=True)
