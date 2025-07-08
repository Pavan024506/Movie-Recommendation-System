import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os

# -------------------------------
# 📥 Load Dataset with Poster URLs
# -------------------------------
df_path = "TeluguMovies_with_Posters.csv"

if not os.path.exists(df_path):
    st.error(f"File not found: {df_path}. Please make sure you've already generated it.")
    st.stop()

# Load data
df = pd.read_csv(df_path)

# Clean data
df.dropna(subset=["Overview"], inplace=True)
df["Genre"] = df["Genre"].fillna("Unknown")
df["Rating"] = df["Rating"].fillna("Unrated")
df["Movie"] = df["Movie"].astype(str).str.strip()
df["lower_title"] = df["Movie"].str.lower()
df["Year"] = df["Year"].fillna(0)
df.reset_index(drop=True, inplace=True)

# Create a combined features column
rating_scaled = df["Rating"].astype(str).str.replace("Unrated", "0")
df["combined_features"] = (
    df["Overview"].astype(str) + " " +
    df["Genre"].astype(str) * 2 + " " +  # Weight Genre
    rating_scaled + " " +
    df["Movie"].astype(str)
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------------
# 🎯 Recommendation Function
# -------------------------------
def recommend_movies(movie_title: str, num_recommendations: int = 5):
    movie_title = movie_title.lower().strip()
    df["clean_title"] = df["Movie"].astype(str).str.lower().str.strip()
    best_match = process.extractOne(movie_title, df["clean_title"])

    if not best_match or best_match[1] < 70:
        return None

    matched_title = best_match[0]
    idx = df[df["clean_title"] == matched_title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores if i[0] != idx][:num_recommendations]

    return df.loc[movie_indices, ["Movie", "Year", "Genre", "Rating", "Poster_URL"]]

# -------------------------------
# 🎨 Streamlit UI
# -------------------------------
st.set_page_config(page_title="🎬 Telugu Movie Recommender", layout="centered")
st.title("🍿 Telugu Movie Recommendation System")

movie_input = st.text_input("Enter a movie name:")
num_recs = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

if st.button("Recommend"):
    if movie_input.strip() == "":
        st.warning("Please enter a movie name.")
    else:
        results = recommend_movies(movie_input, num_recs)
        if results is None or results.empty:
            st.error(f"Movie '{movie_input}' not found in the dataset.")
        else:
            st.success(f"Top {num_recs} recommendations for '{movie_input.title()}':")
            for _, row in results.iterrows():
                year_display = int(row['Year']) if pd.notna(row['Year']) and row['Year'] != 0 else "Unknown"
                st.markdown(f"### 🎬 {row['Movie']} ({year_display})")
                cols = st.columns([1, 3])
                with cols[0]:
                    if pd.notna(row['Poster_URL']):
                        st.image(row['Poster_URL'], width=140)
                    else:
                        st.write("No poster available")
                with cols[1]:
                    st.markdown(
                        f"**Genre:** {row['Genre']}  \n"
                        f"**Rating:** ⭐ {row['Rating']}  \n"
                    )
                st.markdown("---")

# -------------------------------
# 📢 TMDb Attribution
# -------------------------------
st.markdown("---")
st.markdown(
    "#### 📌 Powered by TMDb  \n"
    "*This product uses the TMDb API but is not endorsed or certified by TMDb.*"
)