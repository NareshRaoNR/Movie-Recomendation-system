# -- Similarity score visualization module—
# Load your cleaned dataset (the same one used in the app)
df = movies
# TF-IDF and Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')/
tfidf_matrix = tfidf.fit_transform(df['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

# --- Similarity Score Visualization ---
def show_similar_movies(movie_title, top_n=10):
    matches = difflib.get_close_matches(movie_title, df['title'], n=1)
    if not matches:
        print("No close match found.")
        return

    idx = df[df['title'] == matches[0]].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    titles = [df.iloc[i[0]].title for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    # Print results
    print(f"\nTop {top_n} similar movies to '{movie_title}':\n")
    for title, score in zip(titles, scores):
        print(f"{title} ({score:.3f})")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(titles[::-1], scores[::-1], color='skyblue')
    plt.xlabel('Similarity Score')
    plt.title(f"Top {top_n} Similar Movies to '{movie_title}'")
    plt.tight_layout()
    plt.show()

# Example Usage
show_similar_movies("Iron Man", top_n=10)
