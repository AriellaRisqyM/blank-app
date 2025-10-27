# ==============================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML)
# Pipeline: Preprocessing → Filter Polri → Labeling → TF-IDF → Naive Bayes & SVM
# ==============================================================================

import streamlit as st
import pandas as pd
import requests
import json
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm.auto import tqdm
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

tqdm.pandas()

# ==============================================================================
# 🔹 Fungsi Filter Data Polri
# ==============================================================================
def filter_polri(df, text_column='case_folded_text'):
    """Memfilter data agar hanya yang relevan dengan Polri."""
    st.info("🔍 Memfilter data terkait Polri...")

    keywords_polri = [
        'polri', 'polisi', 'kapolri', 'brimob', 'polda',
        'polsek', 'polres', 'satlantas', 'bhayangkara', 'penyidik'
    ]
    exclude_keywords = [
        'tni', 'tentara', 'prajurit', 'kkb', 'ad', 'au', 'al', 'kostrad', 'kopassus'
    ]

    pattern_polri = r'\b(?:' + '|'.join(keywords_polri) + r')\b'
    pattern_exclude = r'\b(?:' + '|'.join(exclude_keywords) + r')\b'

    if text_column not in df.columns:
        st.warning(f"Kolom '{text_column}' tidak ditemukan, melewati filter Polri.")
        return df

    mask_polri = df[text_column].str.contains(pattern_polri, flags=re.IGNORECASE, na=False)
    mask_exclude = df[text_column].str.contains(pattern_exclude, flags=re.IGNORECASE, na=False)

    df_filtered = df[mask_polri & ~mask_exclude].copy()

    st.success(f"✅ Data setelah filter Polri: {len(df_filtered)} dari {len(df)} baris.")
    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data relevan dengan Polri setelah filter.")
    return df_filtered

# ==============================================================================
# 🔹 Fungsi Preprocessing Dasar
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# 🔹 Fungsi Labeling Lexicon
# ==============================================================================
def label_sentiment_simple(text, positive_lexicon, negative_lexicon):
    if not isinstance(text, str): return 'netral'
    tokens = text.split()
    pos = sum(1 for t in tokens if t in positive_lexicon)
    neg = sum(1 for t in tokens if t in negative_lexicon)
    if pos > neg: return 'positif'
    elif neg > pos: return 'negatif'
    else: return 'netral'

# ==============================================================================
# 🔹 Fungsi Preprocessing + Filter + Labeling
# ==============================================================================
@st.cache_data
def preprocess_filter_label(df, text_col, positive_lexicon, negative_lexicon):
    st.info("⚙️ Memulai Preprocessing + Filter Polri + Labeling...")

    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df.dropna(subset=['cleaned_text'], inplace=True)
    df['case_folded_text'] = df['cleaned_text'].str.lower()

    # Filter Polri
    df = filter_polri(df, text_column='case_folded_text')
    if df.empty:
        return df

    # Labeling
    st.write("🏷️ Melabel data (positif/negatif)...")
    df['sentiment'] = df['case_folded_text'].progress_apply(
        lambda x: label_sentiment_simple(x, positive_lexicon, negative_lexicon)
    )

    st.success(f"✅ Labeling selesai: {df['sentiment'].value_counts().to_dict()}")
    return df[['cleaned_text', 'case_folded_text', 'sentiment']]

# ==============================================================================
# 🔹 Fungsi TF-IDF + Model NB & SVM
# ==============================================================================
def train_models(df, max_features=7000, test_size=0.3):
    st.info("🤖 Melatih model Machine Learning (Naive Bayes & SVM)...")

    X = df['case_folded_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Naive Bayes
    nb = MultinomialNB(alpha=0.2)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)

    # SVM
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)

    results = {
        'nb': {'acc': nb_acc, 'report': classification_report(y_test, nb_pred, output_dict=True)},
        'svm': {'acc': svm_acc, 'report': classification_report(y_test, svm_pred, output_dict=True)},
    }

    st.success(f"✅ Akurasi NB: {nb_acc:.2%}, SVM: {svm_acc:.2%}")
    return results

# ==============================================================================
# 🔹 Load Lexicon
# ==============================================================================
@st.cache_resource
def load_inset_lexicons():
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    pos = set(pd.read_csv(pos_url, sep='\t', header=None)[0].dropna().astype(str))
    neg = set(pd.read_csv(neg_url, sep='\t', header=None)[0].dropna().astype(str))
    st.success(f"Leksikon dimuat: +{len(pos)} positif, -{len(neg)} negatif")
    return pos, neg

# ==============================================================================
# 🔹 Streamlit UI
# ==============================================================================
st.set_page_config(page_title="Analisis Sentimen Polri", layout="wide")
st.title("📊 Analisis Sentimen Polri (Lexicon + Machine Learning)")
st.markdown("**Pipeline:** Preprocessing → Filter Polri → Labeling → TF-IDF → NB & SVM")

pos_lex, neg_lex = load_inset_lexicons()

uploaded_file = st.file_uploader("📂 Unggah file CSV yang berisi teks", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"File berhasil diunggah: {uploaded_file.name} ({len(df)} baris)")

    text_col = st.selectbox("📝 Pilih kolom teks:", df.columns.tolist())

    if st.button("🚀 Jalankan Analisis"):
        with st.spinner("Memproses data..."):
            df_processed = preprocess_filter_label(df, text_col, pos_lex, neg_lex)

        if not df_processed.empty:
            st.dataframe(df_processed.head(10))

            st.subheader("Distribusi Sentimen")
            st.bar_chart(df_processed['sentiment'].value_counts())

            with st.spinner("🔢 Melatih model ML..."):
                results = train_models(df_processed)

            st.subheader("Hasil Model")
            col1, col2 = st.columns(2)
            col1.metric("Naive Bayes", f"{results['nb']['acc']:.2%}")
            col2.metric("SVM (Linear)", f"{results['svm']['acc']:.2%}")

            st.subheader("📥 Unduh Hasil Labeling")
            csv_data = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh CSV",
                data=csv_data,
                file_name="hasil_sentimen_polri.csv",
                mime="text/csv"
            )
        else:
            st.warning("Tidak ada data relevan dengan Polri setelah filter.")

else:
    st.info("Silakan unggah file CSV untuk memulai analisis.")
