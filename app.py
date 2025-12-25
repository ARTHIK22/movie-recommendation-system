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

# ---------------- FETCH MULTIPLE POSTERS ----------------
def fetch_more_posters(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    data = requests.get(url, params=params).json()

    if data.get("results"):
        movie_id = data["results"][0]["id"]
        img_url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
        img_data = requests.get(img_url, params={"api_key": TMDB_API_KEY}).json()

        posters = []
        if img_data.get("posters"):
            for p in img_data["posters"][:5]:  # Show max 5 posters
                posters.append("https://image.tmdb.org/t/p/w500" + p["file_path"])
        return posters
    return []

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


# ---------------- PAGE CONFIG ----------------
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


# ---------------- ML MODEL ----------------
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

# ---------------- LIVE SEARCH SECTION ----------------
st.markdown("### 🔍 Search Any Movie (Live from TMDB)")
search_query = st.text_input("Type movie name...", placeholder="Example: Avatar, Inception, Joker")

def get_movie_from_api(query):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query}
    data = requests.get(url, params=params).json()

    if data.get("results"):
        movie = data["results"][0]
        title = movie.get("title", "Unknown Movie")
        overview = movie.get("overview", "No overview available")
        genre = "General"
        poster = "https://image.tmdb.org/t/p/w500" + movie["poster_path"] if movie.get("poster_path") else None
        
        # Add this movie temporarily to dataset for recommendation calculation
        movies.loc[len(movies)] = [title, overview, genre, overview + " " + genre]
        return title, overview, genre, poster
    
    return None, None, None, None

# Show Searched Movie
if search_query:
    st.info(f"Searching: **{search_query}**...")
    title, overview, genre, poster = get_movie_from_api(search_query)

    if title:
        # Re-train model after adding new movie
        movies["combined"] = movies["overview"] + " " + movies["genres"]
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['combined'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        st.success(f"✔ Movie Found: {title}")
        if poster:
            st.image(poster, width=300)
        st.write("📌 **Overview:** " + overview)
        st.write("🎭 **Genre:** " + genre)

        movie_name = title  

        # Set selected movie for recommendation
        movie_name = title  

# ---------------- UI ----------------
selected_genre = st.selectbox("🎭 Select Genre", ["All"] + sorted(movies['genres'].unique()))
st.markdown("### 🎥 Select from existing list (Optional)")
movie_name_dropdown = st.selectbox("OR choose from database:", movies['title'].values)

# Priority: if search used → use searched movie, else dropdown movie
movie_name = movie_name if 'movie_name' in locals() else movie_name_dropdown

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
            movie = row['title']
            poster = fetch_poster(movie)
            trailer = fetch_trailer(movie)

            with cols[i % 3]:
                if poster:
                    st.image(poster, use_container_width=True)
                st.markdown(f"**{movie}** 🎬")

                # 🎞️ TRAILER BUTTON YAHI AAYEGA
                if trailer:
                    st.markdown(
                        f'<a href="{trailer}" target="_blank">'
                        f'<button style="background:#ff4b4b;color:white;padding:6px 10px;'
                        f'border:none;border-radius:5px;cursor:pointer;">▶ Watch Trailer</button></a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<span style="color:gray;">❌ Trailer not available</span>', unsafe_allow_html=True)


                        # ----- SHOW MORE POSTERS -----
                more = fetch_more_posters(movie)
                if more and isinstance(more, list):
                    st.markdown("<br>📌 More Posters:", unsafe_allow_html=True)
                    img_cols = st.columns(len(more))
                    for pi, p in enumerate(more):
                        with img_cols[pi]:
                            st.image(p, use_container_width=True)


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Made by <b>Arthik Dwivedi</b></center>", unsafe_allow_html=True)


