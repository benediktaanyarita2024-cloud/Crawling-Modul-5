# Analisis Sentimen Tweet Karhutla

Aplikasi Streamlit untuk analisis sentimen tweet tentang kebakaran hutan dan lahan (karhutla), dari crawling Twitter (X) sampai dashboard.

## Alur

| No | File | Isi |
|---|---|---|
| 0 | `0.crawling_colab.ipynb` | Crawling tweet dengan tweet-harvest di Google Colab, lalu merge |
| 1 | `1.cleaning.ipynb` | Membersihkan teks (link, mention, hashtag, angka, simbol) |
| 2 | `2.clustering.ipynb` | TF-IDF, K-Means, silhouette score, visualisasi PCA |
| 3 | `3.predict_sentiment.ipynb` | Pelabelan sentimen dengan model Naive Bayes, word cloud |
| 4 | `4.train_model.ipynb` | Melatih model Logistic Regression |
| - | `dashboard.py` | Aplikasi Streamlit (Home, Dataset & Analysis, KMeans Clustering, About) |

## Dataset

- Keyword: `karhutla`
- 280 tweet hasil crawling, 261 tweet setelah cleaning
- Sentimen: neutral 122, positive 90, negative 49

## Menjalankan

```
pip install -r requirements.txt
streamlit run dashboard.py
```
