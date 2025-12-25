import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------- API KEY ----------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# ---------------- PAGE CONFIG (Netflix Mode) ----------------
st.set_page_config(page_title="Netflix Movie Recommender", page_icon="🎬", layout="wide")

# ---------------- NETFLIX UI CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #000 !important;
    color: #fff !important;
}

.title {
    font-size: 60px;
    font-weight: 800;
    text-align: center;
    color: #E50914;
    text-shadow: 0px 0px 20px #E50914;
}
.subtitle {
    text-align:center;
    font-size: 20px;
    color: #ddd;
    margin-bottom: 40px;
}

.selectbox, .stTextInput>div>div>input {
    background:#111!important; color:white!important;
    border:1px solid #333!important; border-radius:8px!important;
}

.movie-card {
    background: #111;
    border-radius: 10px;
    padding: 10px;
    text-align: center;
    transition: 0.3s;
}
.movie-card:hover {
    transform: scale(1.08);
    box-shadow: 0 0 20px #E50914;
}

button {
    background:#E50914 !important; 
    color:white!important;
    border:none!important;
    border-radius:6px!important;
    padding:8px 14px!important;
    font-weight:600!important;
}
button:hover {
    background:#B20710!important;
}

.play-btn {
    background:#E50914;
    padding:8px 14px;
    border-radius:5px;
    color:white; text-decoration:none;
    font-weight:600;
}
.play-btn:hover {
    background:#b20710;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">NETFLIX - Style Movie Recommender</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="subtitle">AI Based Recommendations With Trailers, Posters & Live Search 🚀</h4>', unsafe_allow_html=True)

# ---------------- FUNCTIONS ----------------
def fetch_poster(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    data = requests.get(url, params={"api_key": TMDB_API_KEY, "query": movie_title}).json()
    if data.get("results") and data["results"][0].get("poster_path"):
        return "https://image.tmdb.org/t/p/w500" + data["results"][0]["poster_path"]
    return None

def fetch_trailer(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    data = requests.get(url, params={"api_key": TMDB_API_KEY, "query": movie_title}).json()
    if not data.get("results"): return None
    movie_id = data["results"][0]["id"]
    videos = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
        params={"api_key": TMDB_API_KEY}
    ).json()
    for v in videos.get("results", []):
        if v["type"] == "Trailer":
            return f"https://www.youtube.com/watch?v={v['key']}"
    return None

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")
if "genres" not in movies.columns: movies["genres"] = "General"
movies["combined"] = movies["overview"] + " " + movies["genres"]
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title, genre, count):
    if movie_title not in movies["title"].values: return []
    idx = movies[movies["title"] == movie_title].index[0]
    sim = sorted(list(enumerate(cosine_sim[idx])), key=lambda x:x[1], reverse=True)[1:count+1]
    recs = movies.iloc[[i[0] for i in sim]]
    if genre != "All":
        recs = recs[recs["genres"].str.contains(genre, case=False)]
    return recs

# ---------------- UI INPUTS ----------------
search = st.text_input("🔍 Search Movie from TMDB", placeholder="Example: Avatar, Joker, Iron Man")
genre = st.selectbox("🎭 Genre Filter", ["All"] + sorted(movies["genres"].unique()))
movie = st.selectbox("🎥 Select Base Movie", movies["title"].values)
count = st.slider("🎯 Number of Recommendations", 3, 15, 6)

# ---------------- RECOMMEND ----------------
if st.button("🚀 Generate Recommendations in Netflix Style"):
    st.markdown("## 🍿 Recommended For You")
    recs = recommend(movie, genre, count)
    cols = st.columns(3)

    for i, row in recs.iterrows():
        title = row["title"]
        poster = fetch_poster(title)
        trailer = fetch_trailer(title)
        with cols[i % 3]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            if poster: st.image(poster, use_container_width=True)
            st.write(f"**{title}**")
            if trailer:
                st.markdown(f'<a class="play-btn" target="_blank" href="{trailer}">▶ Play Trailer</a>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown('<p style="text-align:center;color:#777;">Made by <b>Arthik Dwivedi</b> | Netflix UI Edition 🎬</p>', unsafe_allow_html=True)
