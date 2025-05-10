import streamlit as st
import pandas as pd
import numpy as np
import difflib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load dataset
movies = pd.read_csv('tmdb_5000_movies.csv')

#Data cleaning

# Drop duplicates and rows with missing overviews
movies.drop_duplicates(subset='title', inplace=True)
movies.dropna(subset=['overview'], inplace=True)

# Select relevant columns
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'popularity', 'vote_average', 'vote_count']]

# Helper to extract names from stringified JSON
def extract_names(text):
    try:
        return " ".join([item['name'] for item in ast.literal_eval(text)])
    except:
        return ""

# Apply to 'genres' and 'keywords'
movies['genres'] = movies['genres'].apply(extract_names)
movies['keywords'] = movies['keywords'].apply(extract_names)

# Fill NaNs in 'keywords'
movies['keywords'].fillna("", inplace=True)

# Combine into a single text column for content filtering
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']
movies['tags'] = movies['tags'].str.lower()

# Optionally, drop rows with empty tags
movies = movies[movies['tags'].str.strip() != '']

# --- Simple EDA ---
st.title("AI-Driven Movie Recommendation System")
st.subheader("Dataset Preview and Stats")
if st.checkbox("Show raw data"):
    st.dataframe(movies.head())

st.markdown(f"**Number of Movies**: {movies.shape[0]}")
st.markdown(f"**Columns**: {', '.join(movies.columns)}")

# --- Popularity-Based Recommendation ---
def get_popular_movies(df, top_n=10):
    return df.sort_values(by='popularity', ascending=False).head(top_n)[['title', 'popularity']]

st.subheader("Top 10 Popular Movies")
st.table(get_popular_movies(movies))

# --- Content-Based Recommendation ---
st.subheader("Get Movie Recommendations")

# Combine features
movies['tags'] = movies['genres'] + " " + movies['keywords'] + " " + movies['overview']
movies['tags'] = movies['tags'].astype(str)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# TMDB Poster Helper
def get_movie_details(title, api_key="c7b18deb3171cf125bd4514a58aaa735"):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if idx.empty:
        return None
    movie_id = movies.loc[idx[0], 'id']
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "title": data.get("title"),
            "rating": data.get("vote_average"),
            "poster_url": f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get("poster_path") else None,
            "overview": data.get("overview"),
            "release_date": data.get("release_date"),
            "genres": ", ".join([genre['name'] for genre in data.get("genres", [])])
        }
    return None

def recommend(movie_title):
    try:
        index = difflib.get_close_matches(movie_title, movies['title'], n=1)[0]
        movie_idx = movies[movies['title'] == index].index[0]
        similarity_scores = list(enumerate(cosine_sim[movie_idx]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
        recommended = []
        for i in sorted_movies:
            title = movies.iloc[i[0]].title
            details = get_movie_details(title)
            if details:
                recommended.append(details)
        return recommended
    except:
        return []

user_movie = st.text_input("Enter a movie name:")
if user_movie:
    recommendations = recommend(user_movie)
    if recommendations:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.markdown(f"### {movie['title']}")
            st.markdown(f"*Rating*: {movie['rating']}/10")
            st.markdown(f"*Release Date*: {movie['release_date']}")
            st.markdown(f"*Genres*: {movie['genres']}")
            st.markdown(f"*Overview*: {movie['overview']}")
            if movie['poster_url']:
                st.image(movie['poster_url'], width=250)
            st.markdown("---")
    else:
        st.warning("Movie not found. Try another title.")
