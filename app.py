import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- API KEY ----------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# ---------------- FETCH POSTER ----------------
def fetch_poster(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    data = requests.get(url, params=params).json()
    
    if data.get("results"):
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    return None

# ---------------- FETCH TRAILER ----------------
def fetch_trailer(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    data = requests.get(url, params=params).json()

    if data.get("results"):
        movie_id = data["results"][0]["id"]
        video_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        video_data = requests.get(video_url, params={"api_key": TMDB_API_KEY}).json()

        for video in video_data.get("results", []):
            if video["type"] == "Trailer" and video["site"] == "YouTube":
                return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# ---------------- PAGE CONFIG + TITLE ----------------
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
st.markdown("<h1 style='text-align:center;color:#ff4b4b;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:#ccc;'>AI-based Recommendations with Posters, Genres & Trailers</h4>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")

if "genres" not in movies.columns:
    movies["genres"] = "General"

movies = movies[['title', 'overview', 'genres']]
movies.dropna(inplace=True)
movies["combined"] = movies["overview"] + " " + movies["genres"]

# ---------------- MODEL ----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title, selected_genre, count):
    if movie_title not in movies['title'].values:
        return []
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:count+1]
    recs = movies.iloc[[i[0] for i in sim_scores]]

    if selected_genre != "All":
        recs = recs[recs['genres'].str.contains(selected_genre, case=False, na=False)]
    return recs

# ---------------- UI ----------------
selected_genre = st.selectbox("🎭 Select Genre", ["All"] + sorted(movies['genres'].unique()))
movie_name = st.selectbox("🎥 Select a movie", movies['title'].values)
num_recs = st.slider("📌 How many recommendations do you want?", 3, 10, 5)

# ---------------- RECOMMEND ----------------
if st.button("🚀 Recommend Movies"):
    recs = recommend(movie_name, selected_genre, num_recs)

    if len(recs) == 0:
        st.error("No similar movies found 😕 Try another genre or movie.")
    else:
        st.success(f"🍿 Recommended Movies Based On **{movie_name}**")
        cols = st.columns(3)

        for i, row in recs.iterrows():
            movie = row["title"]
            poster = fetch_poster(movie)
            trailer = fetch_trailer(movie)

            with cols[i % 3]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{movie}** 🎬")

                if trailer:
                    st.markdown(f"[▶ Watch Trailer]({trailer})", unsafe_allow_html=True)
                else:
                    st.markdown("❌ Trailer not found")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Made by <b>Arthik Dwivedi</b></center>", unsafe_allow_html=True)
