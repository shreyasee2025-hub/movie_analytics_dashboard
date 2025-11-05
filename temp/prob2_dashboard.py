# =============================================================
# PROBLEM 2 DASHBOARD (LIVE AZURE VERSION)
# =============================================================

import streamlit as st
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
import re

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")

# Replace with your Azure Cosmos DB URI
MONGO_URI = "mongodb+srv://shreyasee:BigData508@cosmos-shrey-mflix.global.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
DB_NAME = "sample_mflix"

# -------------------------------------------------------------
# CONNECT TO DATABASE
# -------------------------------------------------------------
@st.cache_resource
def connect_db():
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

db = connect_db()
movies_col = db["movies"] if db is not None else None

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def clean_text(text):
    """Remove punctuation and lowercase text for wordclouds & topic modeling"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()

def create_wordcloud(text, title):
    wc = WordCloud(width=1200, height=500, background_color="white").generate(text)
    st.image(wc.to_array(), caption=title, use_column_width=True)

# -------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------
st.title("Movie Analytics Dashboard — Azure Cosmos DB (Mongo API)")
st.caption("Live dashboard connected to Azure-hosted document database")

tabs = st.tabs(["Overview", "EDA", "Time Trends", "Text Analysis", "Topics", "Storytelling"])

# -------------------------------------------------------------
# TAB 1 — OVERVIEW
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("Database Overview")

    try:
        collections = db.list_collection_names()
        st.write("**Available Collections:**", collections)

        # Count basic stats
        count_movies = movies_col.count_documents({})
        st.metric("Total Movies", count_movies)
        sample_doc = movies_col.find_one({}, {"title": 1, "year": 1, "genres": 1, "_id": 0})
        st.write("**Example Document:**")
        st.json(sample_doc)
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# -------------------------------------------------------------
# TAB 2 — EDA
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Exploratory Data Analysis (Genres & Ratings)")

    try:
        pipeline = [
            {"$unwind": "$genres"},
            {"$match": {"genres": {"$nin": [None, "", " "]}}},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}, "avg_rating": {"$avg": "$imdb.rating"}}},
            {"$sort": {"count": -1}}
        ]
        genre_df = pd.DataFrame(list(movies_col.aggregate(pipeline)))

        if not genre_df.empty:
            st.write("**Top 10 Genres by Count:**")
            st.bar_chart(genre_df.set_index("_id")["count"].head(10))

            st.write("**Average Rating by Genre:**")
            st.line_chart(genre_df.set_index("_id")["avg_rating"].head(10))
        else:
            st.warning("No genre data available.")
    except Exception as e:
        st.error(f"EDA Error: {e}")

# -------------------------------------------------------------
# TAB 3 — TIME TRENDS
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Time Trends — Average IMDb Rating per Year")

    try:
        pipeline_year = [
            {"$match": {"year": {"$gte": 1900}, "imdb.rating": {"$ne": None}}},
            {"$group": {"_id": "$year", "avg_rating": {"$avg": "$imdb.rating"}}},
            {"$sort": {"_id": 1}}
        ]
        year_df = pd.DataFrame(list(movies_col.aggregate(pipeline_year)))

        if not year_df.empty:
            year_df.rename(columns={"_id": "Year"}, inplace=True)
            st.line_chart(year_df.set_index("Year")["avg_rating"])
        else:
            st.warning("No yearly rating data available.")
    except Exception as e:
        st.error(f"Time trend error: {e}")

# -------------------------------------------------------------
# TAB 4 — TEXT ANALYSIS
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Text Analysis — WordClouds & Common Phrases")

    try:
        docs = list(movies_col.find({}, {"plot": 1, "title": 1, "_id": 0}).limit(3000))
        df = pd.DataFrame(docs)

        if not df.empty:
            df["text"] = (df["title"].astype(str) + " " + df["plot"].astype(str)).apply(clean_text)
            text = " ".join(df["text"].tolist())

            create_wordcloud(text, "Most Common Words in Titles + Plots")

            vec = CountVectorizer(ngram_range=(2, 2), stop_words="english", min_df=5)
            X = vec.fit_transform(df["text"])
            freqs = np.asarray(X.sum(axis=0)).ravel()
            bigrams = np.array(vec.get_feature_names_out())
            top_idx = freqs.argsort()[::-1][:10]
            bigram_df = pd.DataFrame({"Bigram": bigrams[top_idx], "Frequency": freqs[top_idx]})
            st.table(bigram_df)
        else:
            st.warning("No text data found for analysis.")
    except Exception as e:
        st.error(f"Text analysis error: {e}")

# -------------------------------------------------------------
# TAB 5 — TOPIC MODELING
# -------------------------------------------------------------
with tabs[4]:
    st.subheader("Topic Modeling — Discover Hidden Themes (NMF)")

    try:
        docs = list(movies_col.find({}, {"plot": 1, "_id": 0}).limit(2000))
        df = pd.DataFrame(docs)
        corpus = df["plot"].dropna().apply(clean_text).tolist()

        if len(corpus) > 50:
            tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
            X = tfidf.fit_transform(corpus)

            nmf = NMF(n_components=5, random_state=42, max_iter=400)
            nmf.fit(X)
            terms = np.array(tfidf.get_feature_names_out())

            for i, comp in enumerate(nmf.components_):
                top_words = [terms[j] for j in comp.argsort()[-10:][::-1]]
                st.write(f"**Topic {i+1}:**", ", ".join(top_words))
        else:
            st.warning("Not enough text data for topic modeling.")
    except Exception as e:
        st.error(f"Topic modeling error: {e}")

# -------------------------------------------------------------
# TAB 6 — STORYTELLING / NARRATIVE
# -------------------------------------------------------------
with tabs[5]:
    st.subheader("Narrative — Analytical Purpose and Context")
    st.markdown("""
    This dashboard connects to a **document database hosted on Azure Cosmos DB (for MongoDB)** containing movie data.
    
    It demonstrates how cloud-based NoSQL data can be analyzed and visualized using Streamlit.
    The analysis includes:
    - Genre and rating distributions  
    - Yearly rating trends  
    - Text pattern extraction (WordClouds)  
    - Topic modeling to identify thematic clusters  
    
    **Business / Social Value:**  
    Such dashboards are vital for **education**, **media analytics**, or **IoT data storytelling**.
    They enable interactive exploration of unstructured data directly from a cloud database, combining real-time access and rich visualization.
    """)
