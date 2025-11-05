# =============================================================
# AZURE MOVIE ANALYTICS DASHBOARD — Streamlit Cloud Version
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
st.set_page_config(
    page_title="Azure Movie Analytics Dashboard",
    layout="wide"
)

# Light theme visual configuration
plt.style.use("default")
plt.rcParams.update({
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "text.color": "#000000"
})

# Distinct color palette for headings
heading_colors = {
    "main": "#0052cc",      # Deep Azure blue
    "overview": "#d35400",  # Burnt orange
    "eda": "#16a085",       # Teal green
    "time": "#8e44ad",      # Royal purple
    "text": "#c0392b",      # Crimson red
    "topic": "#27ae60",     # Emerald green
    "story": "#2980b9"      # Bright blue
}

chart_colors = ["#0066cc", "#009999", "#ff6600", "#9933ff", "#33cc33"]

# -------------------------------------------------------------
# MONGO CONNECTION (via Streamlit Secrets)
# -------------------------------------------------------------
MONGO_URI = st.secrets.get("MONGO_URI", "")
DB_NAME = st.secrets.get("DB_NAME", "sample_mflix")

@st.cache_resource
def connect_db():
    """Connect to Azure Cosmos DB (Mongo API)."""
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
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()

def create_wordcloud(text, title):
    if not text or len(text.strip()) == 0:
        st.warning("⚠️ Not enough text data to generate a WordCloud.")
        return
    try:
        wc = WordCloud(width=1200, height=500, background_color="white", colormap="Set2")
        wc.generate(text)
        st.image(wc.to_array(), caption=title, use_column_width=True)
    except Exception as e:
        st.error(f"WordCloud generation failed: {e}")

def df_download_button(df, filename, label="📥 Download CSV"):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=f"{filename}.csv", mime="text/csv")

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.header("📊 Dashboard Navigation")
menu = st.sidebar.radio(
    "Select Section:",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Time Trends",
        "Text Analysis",
        "Topic Modeling",
        "Storytelling / Narrative"
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by Shreyasee Poddar, DSA 508 (Big Data Platforms)**")
st.sidebar.markdown("*Powered by Azure Cosmos DB & Streamlit*")

# -------------------------------------------------------------
# HEADER SECTION
# -------------------------------------------------------------
st.markdown(
    f"<h1 style='color:{heading_colors['main']}; font-weight:800;'>Azure Movie Analytics Dashboard</h1>",
    unsafe_allow_html=True
)
st.caption("Interactive analysis of movie data stored in Azure Cosmos DB (Mongo API)")

# -------------------------------------------------------------
# METRIC CARDS
# -------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    total = movies_col.count_documents({}) if movies_col is not None else 0
    st.markdown(f"<h4 style='color:{heading_colors['overview']}'>🎞️ Total Movies</h4>", unsafe_allow_html=True)
    st.metric("", f"{total:,}")
with col2:
    unique_genres = len(movies_col.distinct("genres")) if movies_col is not None else 0
    st.markdown(f"<h4 style='color:{heading_colors['eda']}'>🎭 Unique Genres</h4>", unsafe_allow_html=True)
    st.metric("", unique_genres)
with col3:
    try:
        avg_rating = movies_col.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$imdb.rating"}}}])
        avg = round(list(avg_rating)[0]["avg"], 2) if movies_col is not None else 0
        st.markdown(f"<h4 style='color:{heading_colors['time']}'>⭐ Average IMDb Rating</h4>", unsafe_allow_html=True)
        st.metric("", avg)
    except:
        st.metric("Average IMDb Rating", "N/A")

st.markdown("---")

# -------------------------------------------------------------
# PAGE LOGIC
# -------------------------------------------------------------
if menu == "Overview":
    st.markdown(f"<h2 style='color:{heading_colors['overview']}'>Database Overview</h2>", unsafe_allow_html=True)
    try:
        collections = db.list_collection_names()
        st.write("**Available Collections:**", collections)
        st.subheader("Sample Document:")
        sample_doc = movies_col.find_one({}, {"title": 1, "year": 1, "genres": 1, "imdb": 1, "_id": 0})
        st.json(sample_doc)
        st.success("✅ Connected successfully to Azure Cosmos DB!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

elif menu == "Exploratory Data Analysis":
    st.markdown(f"<h2 style='color:{heading_colors['eda']}'>Exploratory Data Analysis — Genres & Ratings</h2>", unsafe_allow_html=True)
    try:
        pipeline = [
            {"$unwind": "$genres"},
            {"$match": {"genres": {"$nin": [None, "", " "]}}},
            {"$group": {"_id": "$genres", "count": {"$sum": 1}, "avg_rating": {"$avg": "$imdb.rating"}}},
            {"$sort": {"count": -1}}
        ]
        genre_df = pd.DataFrame(list(movies_col.aggregate(pipeline)))
        if not genre_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(genre_df["_id"].head(10), genre_df["count"].head(10), color=heading_colors["eda"])
            ax.set_title("Top 10 Genres by Movie Count")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            df_download_button(genre_df, "genres_summary", "Download Genre Data (CSV)")
        else:
            st.warning("No genre data found.")
    except Exception as e:
        st.error(f"EDA error: {e}")

elif menu == "Time Trends":
    st.markdown(f"<h2 style='color:{heading_colors['time']}'>Time Trends — IMDb Ratings Over Time</h2>", unsafe_allow_html=True)
    try:
        pipeline_year = [
            {"$match": {"year": {"$gte": 1900}, "imdb.rating": {"$ne": None}}},
            {"$group": {"_id": "$year", "avg_rating": {"$avg": "$imdb.rating"}}},
            {"$sort": {"_id": 1}}
        ]
        year_df = pd.DataFrame(list(movies_col.aggregate(pipeline_year)))
        if not year_df.empty:
            year_df.rename(columns={"_id": "Year"}, inplace=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(year_df["Year"], year_df["avg_rating"], color=heading_colors["time"], linewidth=1.2)
            ax.set_title("Average IMDb Rating per Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Rating")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            df_download_button(year_df, "yearly_ratings", "Download Yearly Trends (CSV)")
        else:
            st.warning("No yearly data.")
    except Exception as e:
        st.error(f"Time trend error: {e}")

elif menu == "Text Analysis":
    st.markdown(f"<h2 style='color:{heading_colors['text']}'>Text Analysis — WordCloud & Common Phrases</h2>", unsafe_allow_html=True)
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
            st.dataframe(bigram_df)
            df_download_button(bigram_df, "common_bigrams", "Download Text Analysis (CSV)")
        else:
            st.warning("No text data available.")
    except Exception as e:
        st.error(f"Text analysis error: {e}")

elif menu == "Topic Modeling":
    st.markdown(f"<h2 style='color:{heading_colors['topic']}'>Topic Modeling — Discover Hidden Themes (NMF)</h2>", unsafe_allow_html=True)
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
            st.markdown(f"<h4 style='color:{heading_colors['topic']};'>Discovered Topics:</h4>", unsafe_allow_html=True)
            topic_words_list = []
            for i, comp in enumerate(nmf.components_):
                top_words = [str(terms[j]) for j in comp.argsort()[-10:][::-1]]
                topic_words_list.append(", ".join(top_words))
                st.markdown(
                    f"<b style='color:{heading_colors['topic']}'>Topic {i+1}:</b> "
                    + ", ".join(top_words),
                    unsafe_allow_html=True
                )
            topic_df = pd.DataFrame({
                "Topic": [f"Topic {i+1}" for i in range(len(topic_words_list))],
                "Top Words": topic_words_list
            })
            df_download_button(topic_df, "topics", "📥 Download Topics (CSV)")
        else:
            st.warning("Not enough data for topic modeling.")
    except Exception as e:
        st.error(f"Topic modeling error: {e}")

elif menu == "Storytelling / Narrative":
    st.markdown(f"<h2 style='color:{heading_colors['story']}'>Narrative — Analytical Purpose and Context</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='color:black; font-size:16px;'>
    The <b style='color:{heading_colors["main"]}'>Azure Movie Analytics Dashboard</b> demonstrates how a 
    <b style='color:{heading_colors["story"]}'>document-oriented cloud database</b> (Azure Cosmos DB for MongoDB)
    can power real-time analytics using <b>Streamlit</b> visualizations.

    <br><br>
    <b style='color:{heading_colors["eda"]}'>🎯 Purpose:</b><br>
    Transform unstructured movie data into meaningful insights — genres, trends, and text-based themes —
    enabling dynamic exploration without writing queries.

    <br><br>
    <b style='color:{heading_colors["time"]}'>📊 Analytical Value:</b><br>
    - Genre and rating distributions  
    - IMDb trends across decades  
    - Common word and phrase analysis  
    - Topic discovery using NMF  

    <br><br>
    <b style='color:{heading_colors["topic"]}'>💡 Decision-Making Impact:</b><br>
    - <b>Retail/Streaming:</b> Identify popular genres and themes for recommendations.  
    - <b>Education:</b> Demonstrate applied Big Data and ML techniques.  
    - <b>IoT/Cloud:</b> Showcase scalable NoSQL-driven analytics.

    <br><br>
    This integrates <b>Azure Cosmos DB</b> with <b>Streamlit</b> for a professional, data-driven storytelling experience.
    </div>
    """, unsafe_allow_html=True)
