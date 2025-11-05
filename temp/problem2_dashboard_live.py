# =============================================================
# PROBLEM 2 DASHBOARD — LIGHT THEME WITH COLORED HEADINGS
# =============================================================
# Run with: streamlit run problem2_dashboard_live.py
# -------------------------------------------------------------

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
    page_title="🎬 Azure Movie Analytics Dashboard",
    page_icon="🎥",
    layout="wide"
)

plt.style.use("default")
plt.rcParams.update({
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "text.color": "#000000"
})

# Color palette and accent colors
chart_colors = ["#0066cc", "#009999", "#ff6600", "#9933ff", "#33cc33"]
accents = {
    "blue": "#0066cc",
    "teal": "#009999",
    "orange": "#ff6600",
    "purple": "#9933ff",
    "green": "#33cc33"
}

# -------------------------------------------------------------
# AZURE CONNECTION
# -------------------------------------------------------------
MONGO_URI = (
    "mongodb+srv://shreyasee:BigData508@cosmos-shrey-mflix.global.mongocluster.cosmos.azure.com/"
    "?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
)
DB_NAME = "sample_mflix"

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
    st.download_button(
        label=label,
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv",
        use_container_width=True
    )

def colored_title(title, color):
    """Render colored section titles."""
    st.markdown(f"<h3 style='color:{color}; font-weight:700;'>{title}</h3>", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.header("📁 Dashboard Navigation")
menu = st.sidebar.radio(
    "Select Section:",
    [
        "🏠 Overview",
        "📊 Exploratory Data Analysis",
        "📈 Time Trends",
        "☁️ Text Analysis",
        "🧩 Topic Modeling",
        "🧾 Storytelling / Narrative"
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("👩‍💻 **Developed by Shreyasee — DSA 508 (Big Data Platforms)**")
st.sidebar.markdown("🌐 *Powered by Azure Cosmos DB & Streamlit*")

# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown(
    f"<h1 style='color:{accents['blue']}; font-weight:800;'>🎬 Azure Movie Analytics Dashboard</h1>",
    unsafe_allow_html=True
)
st.caption("Interactive analysis of movie data stored in Azure Cosmos DB (Mongo API)")

# -------------------------------------------------------------
# METRIC CARDS (with subtle colorized titles)
# -------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    total = movies_col.count_documents({}) if movies_col is not None else 0
    st.markdown(f"<h4 style='color:{accents['blue']}'>🎞️ Total Movies</h4>", unsafe_allow_html=True)
    st.metric("", f"{total:,}")
with col2:
    unique_genres = len(movies_col.distinct("genres")) if movies_col is not None else 0
    st.markdown(f"<h4 style='color:{accents['teal']}'>🎭 Unique Genres</h4>", unsafe_allow_html=True)
    st.metric("", unique_genres)
with col3:
    try:
        avg_rating = movies_col.aggregate([{"$group": {"_id": None, "avg": {"$avg": "$imdb.rating"}}}])
        avg = round(list(avg_rating)[0]["avg"], 2)
        st.markdown(f"<h4 style='color:{accents['orange']}'>⭐ Average IMDb Rating</h4>", unsafe_allow_html=True)
        st.metric("", avg)
    except:
        st.metric("", "N/A")

st.markdown("---")

# -------------------------------------------------------------
# MAIN LOGIC PER PAGE
# -------------------------------------------------------------
if menu == "🏠 Overview":
    colored_title("Database Overview", accents["blue"])
    try:
        collections = db.list_collection_names()
        st.write("**Available Collections:**", collections)
        st.subheader("Sample Document:")
        sample_doc = movies_col.find_one({}, {"title": 1, "year": 1, "genres": 1, "imdb": 1, "_id": 0})
        st.json(sample_doc)
        st.success("✅ Connected successfully to Azure Cosmos DB!")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# -------------------------------------------------------------
elif menu == "📊 Exploratory Data Analysis":
    colored_title("Exploratory Data Analysis — Genres & Ratings", accents["teal"])
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
            ax.bar(genre_df["_id"].head(10), genre_df["count"].head(10), color=accents["blue"])
            ax.set_title("Top 10 Genres by Movie Count", color="black")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
            df_download_button(genre_df, "genres_summary", "📥 Download Genre Data (CSV)")
        else:
            st.warning("No genre data found.")
    except Exception as e:
        st.error(f"EDA error: {e}")

# -------------------------------------------------------------
elif menu == "📈 Time Trends":
    colored_title("Time Trends — IMDb Ratings Over Time", accents["orange"])
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
            ax.plot(year_df["Year"], year_df["avg_rating"], color=accents["orange"], linewidth=1.2)
            ax.set_title("Average IMDb Rating per Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Rating")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            df_download_button(year_df, "yearly_ratings", "📥 Download Yearly Trends (CSV)")
        else:
            st.warning("No yearly data.")
    except Exception as e:
        st.error(f"Time trend error: {e}")

# -------------------------------------------------------------
elif menu == "☁️ Text Analysis":
    colored_title("Text Analysis — WordCloud & Common Phrases", accents["purple"])
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
            df_download_button(bigram_df, "common_bigrams", "📥 Download Text Analysis (CSV)")
        else:
            st.warning("No text data available.")
    except Exception as e:
        st.error(f"Text analysis error: {e}")

# -------------------------------------------------------------
elif menu == "🧩 Topic Modeling":
    colored_title("Topic Modeling — Discover Hidden Themes (NMF)", accents["green"])
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
            topic_words = []
            for i, comp in enumerate(nmf.components_):
                top_words = [terms[j] for j in comp.argsort()[-10:][::-1]]
                st.markdown(f"<b style='color:{accents['green']}'>Topic {i+1}:</b> " + ", ".join(top_words), unsafe_allow_html=True)
                topic_words.append({"Topic": i + 1, "Top Words": ", ".join(top_words)})
            topic_df = pd.DataFrame(topic_words)
            df_download_button(topic_df, "topics", "📥 Download Topics (CSV)")
        else:
            st.warning("Not enough data for topic modeling.")
    except Exception as e:
        st.error(f"Topic modeling error: {e}")

# -------------------------------------------------------------
elif menu == "🧾 Storytelling / Narrative":
    colored_title("Narrative — Analytical Purpose and Context", accents["blue"])
    st.markdown(f"""
    <div style='color:black;'>
    This dashboard connects to a <b style='color:{accents["blue"]}'>document database hosted on Azure Cosmos DB</b> (for MongoDB) containing movie data.
    
    It demonstrates how cloud-based NoSQL data can be analyzed and visualized interactively in Streamlit.
    
    <h4 style='color:{accents["teal"]}'>📊 Insights:</h4>
    - Genre and rating distributions  
    - Yearly IMDb rating trends  
    - Text pattern extraction (WordClouds)  
    - Topic modeling for theme discovery  
    
    <h4 style='color:{accents["orange"]}'>🌍 Real-world Value:</h4>
    - Supports <b>education</b>, <b>media analytics</b>, and <b>IoT storytelling</b>  
    - Combines real-time cloud access with elegant, interactive visuals  
    </div>
    """, unsafe_allow_html=True)
