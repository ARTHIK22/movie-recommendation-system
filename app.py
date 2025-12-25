import requests
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- API KEY ----------------
TMDB_API_KEY = "96c7cb57d4c296204ed0156831e0b5c3"# <-- Yaha apni real key daalo

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
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Movie Recommendations with Posters</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("movies.csv")
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

# ---------------- ML MODEL ----------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# ---------------- UI ----------------
st.markdown("### 🎥 Select a movie")
movie_name = st.selectbox("", movies['title'].values)

if st.button("🚀 Recommend Movies"):
    with st.spinner("Finding the best movies for you..."):
        recs = recommend(movie_name)

    if len(recs) == 0:
        st.error("Movie not found in database!")
    else:
        st.success("🍿 Your Movie Recommendations")

        cols = st.columns(3)
        for i, movie in enumerate(recs):
            poster = fetch_poster(movie)
            with cols[i % 3]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{movie}**")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Made by <b>Arthik Dwivedi</b></center>", unsafe_allow_html=True)
