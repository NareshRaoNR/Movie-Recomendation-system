import streamlit as st  
import pandas as pd  
import numpy as np  
import difflib  
import requests  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
import ast  
  
# --- Data Loading and Cleaning ---  
@st.cache_data  
def load_data():  
    df = pd.read_csv('tmdb_5000_movies.csv')  
    df.drop_duplicates(subset='title', inplace=True)  
    df.dropna(subset=['overview'], inplace=True)  
    df = df[['id', 'title', 'overview', 'genres', 'keywords', 'popularity', 'vote_average', 'vote_count']]  
    df['genres'] = df['genres'].apply(lambda x: " ".join([item['name'] for item in ast.literal_eval(x)]) if pd.notnull(x) else "")  
    df['keywords'] = df['keywords'].apply(lambda x: " ".join([item['name'] for item in ast.literal_eval(x)]) if pd.notnull(x) else "")  
    df['tags'] = (df['overview'] + " " + df['genres'] + " " + df['keywords']).str.lower()  
    df = df[df['tags'].str.strip() != '']  
    return df  
  
movies = load_data()  
  
# --- TF-IDF and Similarity ---  
@st.cache_resource  
def compute_similarity(df):  
    tfidf = TfidfVectorizer(stop_words='english')  
    tfidf_matrix = tfidf.fit_transform(df['tags'])  
    return cosine_similarity(tfidf_matrix)  
  
cosine_sim = compute_similarity(movies)  
  
# --- TMDB API Helper ---  
@st.cache_data(show_spinner=False)  
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
  
# --- Recommendation Function ---  
def recommend(movie_title):  
    matches = difflib.get_close_matches(movie_title, movies['title'], n=1)  
    if not matches:  
        return []  
    movie_idx = movies[movies['title'] == matches[0]].index[0]  
    similarity_scores = list(enumerate(cosine_sim[movie_idx]))  
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]  
    recommended = []  
    for i in sorted_movies:  
        title = movies.iloc[i[0]].title  
        details = get_movie_details(title)  
        if details:  
            recommended.append(details)  
    return recommended  
  
# --- Streamlit Layout ---  
st.title("AI-Driven Movie Recommendation System")  
st.markdown("This app recommends movies based on your input using TF-IDF and TMDB data.")  
  
# --- Dataset Preview ---  
st.subheader("Dataset Preview and Stats")  
if st.checkbox("Show raw data"):  
    st.dataframe(movies.head())  
  
st.markdown(f"*Number of Movies*: {movies.shape[0]}")  
st.markdown(f"*Columns*: {', '.join(movies.columns)}")  
  
# --- Top Movies Section ---  
st.subheader("Top Movies")  
sort_by = st.radio("Sort by:", ["popularity", "vote_average"], horizontal=True)  
all_genres = sorted(set(" | ".join(movies['genres']).split()))  
selected_genre = st.selectbox("Filter by Genre", ["All"] + all_genres)  
top_view_mode = st.radio("View Style:", ["Grid", "List"], horizontal=True)  
  
def get_top_movies(df, sort_col='popularity', genre=None, top_n=10):  
    if genre and genre != "All":  
        df = df[df['genres'].str.contains(genre, case=False)]  
    return df.sort_values(by=sort_col, ascending=False).head(top_n)[['title', sort_col]]  
  
top_movies = get_top_movies(movies, sort_col=sort_by, genre=selected_genre)  
  
if top_view_mode == "Grid":  
    cols = st.columns(5)  
    for idx, (_, row) in enumerate(top_movies.iterrows()):  
        details = get_movie_details(row['title'])  
        if details:  
            col = cols[idx % 5]  
            with col:  
                if details['poster_url']:  
                    st.image(details['poster_url'],width=150)  
                st.markdown(f"{details['title']}")  
                st.caption(f"{details['rating']}/10 | {details['genres']}")  
else:  
    for _, row in top_movies.iterrows():  
        details = get_movie_details(row['title'])  
        if details:  
            st.markdown(f"### {details['title']}")  
            if details['poster_url']:  
                st.image(details['poster_url'], width=150)  
            st.markdown(f"Rating: {details['rating']}")  
            st.markdown(f"Genres: {details['genres']}")  
            st.markdown("---")  
  
# --- Recommendation Section ---  
st.subheader("Get Movie Recommendations")  
movie_list = [""] + movies['title'].sort_values().tolist()  
user_movie = st.selectbox("Select a movie to get recommendations:", movie_list)  
rec_view_mode = st.radio("Recommendations View:", ["Grid", "List"], horizontal=True)  
  
if user_movie != "Select a movie...":  
    with st.spinner("Finding recommendations..."):  
        recommendations = recommend(user_movie)  
    if recommendations:  
        st.subheader("Recommended Movies:")  
        if rec_view_mode == "Grid":  
            cols = st.columns(5)  
            for idx, movie in enumerate(recommendations):  
                col = cols[idx % 5]  
                with col:  
                    if movie['poster_url']:  
                        st.image(movie['poster_url'], width=150)  
                    st.markdown(f"{movie['title']}")  
                    st.caption(f"{movie['rating']}/10 | {movie['release_date']}")  
        else:  
            for movie in recommendations:  
                st.markdown(f"### {movie['title']}")  
                st.markdown(f"Rating: {movie['rating']}/10")  
                st.markdown(f"Release Date: {movie['release_date']}")  
                st.markdown(f"Genres: {movie['genres']}")  
                st.markdown(f"Overview: {movie['overview']}")  
                if movie['poster_url']:  
                    st.image(movie['poster_url'], width=150)  
                st.markdown("---")  
    else:  
        st.warning("No recommendations found. Try another title.")  
else:  
    st.info("Please select a movie from the dropdown to see recommendations.")
