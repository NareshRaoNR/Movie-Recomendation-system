#--EDA PROCESS MODULE—
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from wordcloud import WordCloud

# Load the dataset
df = movies 

# Basic info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated(subset='title').sum())

# Drop duplicates and rows with missing overview
df = df.drop_duplicates(subset='title')
df = df.dropna(subset=['overview'])

# --- UNIVARIATE ANALYSIS ---
plt.figure(figsize=(8, 5))
sns.histplot(df['popularity'], bins=50, kde=True)
plt.title('Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['vote_average'], bins=20, kde=True, color='orange')
plt.title('Vote Average Distribution')
plt.xlabel('Average Rating')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['vote_count'], bins=50, kde=True, color='green')
plt.title('Vote Count Distribution')
plt.xlabel('Vote Count')
plt.ylabel('Count')
plt.show()

# --- FEATURE PROCESSING FOR GENRES AND KEYWORDS ---
def extract_names(text):
    try:
        return " ".join([d['name'] for d in ast.literal_eval(text)])
    except:
        return ""

df['genres_clean'] = df['genres'].apply(extract_names)
df['keywords_clean'] = df['keywords'].apply(extract_names)

# Top genres
all_genres = " ".join(df['genres_clean']).split()
genre_freq = pd.Series(all_genres).value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=genre_freq.values, y=genre_freq.index, palette='viridis')
plt.title('Top 10 Genres')
plt.xlabel('Frequency')
plt.ylabel('Genre')
plt.show()

# --- BIVARIATE ANALYSIS ---
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='popularity', y='vote_count', alpha=0.5)
plt.title('Popularity vs. Vote Count')
plt.xlabel('Popularity')
plt.ylabel('Vote Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='popularity', y='vote_average', alpha=0.5, color='purple')
plt.title('Popularity vs. Vote Average')
plt.xlabel('Popularity')
plt.ylabel('Average Vote')
plt.show()

# --- TEXT LENGTH IN OVERVIEW ---
df['overview_len'] = df['overview'].apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(df['overview_len'], bins=50, kde=True, color='brown')
plt.title('Overview Length Distribution')
plt.xlabel('Character Count')
plt.ylabel('Number of Movies')
plt.show()

# --- WORD CLOUD FOR TAGS (overview + genres + keywords) ---
df['tags'] = (df['overview'] + " " + df['genres_clean'] + " " + df['keywords_clean']).str.lower()

text_blob = " ".join(df['tags'])
wordcloud = WordCloud(width=1000, height=500, background_color='black', max_words=100).generate(text_blob)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Movie Tags', fontsize=20)
plt.show()