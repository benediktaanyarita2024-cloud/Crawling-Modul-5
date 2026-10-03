import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import nltk

# ===================== Konfigurasi =====================
KEYWORD = "karhutla"                               # samakan dengan keyword crawling
DATASET_PATH = f"data/{KEYWORD}_predicted.csv"     # hasil notebook 3
MODEL_PATH = "model/model_lr.joblib"               # hasil notebook 4
TRUE_K = 8                                         # samakan dengan true_k di notebook 2

st.set_page_config(page_title="Sentiment Analysis Karhutla", page_icon="🔥")


# ===================== Load data & model =====================
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH).dropna(subset=["full_text"])


@st.cache_data
def get_stopwords():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as sw
    words = set(sw.words("indonesian")) | set(sw.words("english"))
    words |= {"ya", "yg", "ga", "gak", "ngga", "nggak", "engga", "enggak", "yuk", "dah",
              "aja", "sih", "nih", "tuh", "dong", "deh", "kok", "lah", "nya", "kalo",
              "gue", "gw", "lu", "lo", "udah", "jd", "dgn", "utk", "tdk", "sm", "sj",
              "bang", "pak", "amp", KEYWORD}
    return sorted(words)


pipeline = load_pipeline()


# ===================== Fungsi bantu =====================
def analyze_token_sentiment(docx, pipeline):
    """Prediksi sentimen tiap kata pada kalimat input."""
    pos_list, neg_list, neu_list = [], [], []
    for word in docx.split():
        prediction = pipeline.predict([word])[0]
        proba_max = round(float(pipeline.predict_proba([word]).max()), 3)
        if prediction == "positive":
            pos_list.append([word, proba_max])
        elif prediction == "negative":
            neg_list.append([word, proba_max])
        else:
            neu_list.append(word)
    return {"positives": pos_list, "negatives": neg_list, "neutral": neu_list}


def generate_wordcloud(data, stopwords):
    data_text = " ".join(data["full_text"].astype(str).tolist())
    wc = WordCloud(stopwords=stopwords, background_color="black", max_words=500,
                   width=800, height=400).generate(data_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def plot_top_words(data, stopwords):
    stop = set(stopwords)
    text = " ".join(data["full_text"].astype(str).tolist())
    words = [w for w in text.split() if w not in stop and len(w) > 2]
    top_words = Counter(words).most_common(12)
    kata, counts = zip(*top_words)

    colors = plt.cm.Paired(range(len(kata)))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(kata, counts, color=colors)
    ax.set_xlabel("Kata")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Kata yang sering muncul")
    ax.set_xticks(range(len(kata)))
    ax.set_xticklabels(kata, rotation=45)
    for bar, num in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, num + 0.5, str(num), ha="center")
    st.pyplot(fig)


@st.cache_data
def run_kmeans(texts, stopwords, true_k):
    vectorizer = TfidfVectorizer(stop_words=list(stopwords), min_df=2)
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=10, random_state=42)
    model.fit(X)

    terms = vectorizer.get_feature_names_out()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    cluster_terms = [
        f"Cluster {i} ({(model.labels_ == i).sum()} tweet): "
        + ", ".join(terms[ind] for ind in order_centroids[i, :10])
        for i in range(true_k)
    ]
    score = silhouette_score(X, model.labels_)

    pca = PCA(n_components=2, random_state=0)
    reduced_features = pca.fit_transform(X.toarray())
    reduced_centers = pca.transform(model.cluster_centers_)
    return cluster_terms, score, reduced_features, reduced_centers, model.labels_


# ===================== Halaman =====================
def main():
    st.title("Sentiment Analysis NLP App")
    st.subheader("Analisis Sentimen Tweet Karhutla")

    menu = ["Home", "Dataset & Analysis", "KMeans Clustering", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # ---------- HOME ----------
    if choice == "Home":
        st.subheader("Home")
        with st.form(key="nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label="Analyze")

        col1, col2 = st.columns(2)
        if submit_button and raw_text.strip():
            with col1:
                st.info("Results")
                sentiment = pipeline.predict([raw_text])[0]
                proba = pipeline.predict_proba([raw_text]).max()
                st.write(f"Predicted Sentiment: **{sentiment}** (keyakinan {proba:.0%})")
                if sentiment == "positive":
                    st.markdown("## Positive 😃")
                elif sentiment == "negative":
                    st.markdown("## Negative 😡")
                else:
                    st.markdown("## Neutral 😐")
            with col2:
                st.info("Token Sentiment")
                st.json(analyze_token_sentiment(raw_text, pipeline))
        elif submit_button:
            st.warning("Masukkan kalimat terlebih dahulu.")

    # ---------- DATASET & ANALYSIS ----------
    elif choice == "Dataset & Analysis":
        st.subheader("Dataset & Sentiment Analysis")
        data = load_dataset()
        stopwords = get_stopwords()

        st.write("### Dataset")
        st.dataframe(data[["full_text", "sentiment"]])
        st.caption(f"Jumlah tweet: {len(data)}")

        st.write("### Distribusi Sentimen")
        sentiment_counts = data["sentiment"].value_counts()
        st.bar_chart(sentiment_counts)

        st.write("### Jumlah Sentimen")
        st.dataframe(pd.DataFrame({
            "jumlah": sentiment_counts,
            "persen": (sentiment_counts / len(data) * 100).round(1).astype(str) + " %",
        }))

        st.write("### Word Cloud")
        generate_wordcloud(data, stopwords)

        st.write("### Most Frequent Words")
        plot_top_words(data, stopwords)

    # ---------- KMEANS CLUSTERING ----------
    elif choice == "KMeans Clustering":
        st.subheader("KMeans Clustering Analysis")
        data = load_dataset()
        stopwords = get_stopwords()

        cluster_terms, score, reduced_features, reduced_centers, labels = run_kmeans(
            tuple(data["full_text"].astype(str)), tuple(stopwords), TRUE_K)

        st.write("### Cluster Terms")
        st.text("\n".join(cluster_terms))

        st.write(f"### Silhouette Score: {score:.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1],
                             c=labels, cmap="viridis", s=18, alpha=0.7)
        ax.scatter(reduced_centers[:, 0], reduced_centers[:, 1],
                   marker="x", s=150, c="red", label="Centroids")
        ax.set_title("KMeans Clustering Visualization")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend()
        fig.colorbar(scatter, label="Cluster")
        st.pyplot(fig)

    # ---------- ABOUT ----------
    else:
        st.subheader("About")
        st.write(
            "Aplikasi ini menganalisis sentimen tweet tentang kebakaran hutan dan lahan "
            "(karhutla) yang diambil dari Twitter (X) menggunakan tweet-harvest. "
            "Prediksi sentimen menggunakan model Logistic Regression yang dilatih dari "
            "dataset hasil crawling, sedangkan pengelompokan topik menggunakan K-Means."
        )


if __name__ == "__main__":
    main()
