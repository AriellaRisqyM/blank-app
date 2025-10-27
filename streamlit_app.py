# ==============================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML)
# Pipeline: Preprocessing → Filter Polri → Labeling (2 kelas) → TF-IDF → NB & SVM
# Dua Fitur: Upload File & Analisis Teks Input
# ==============================================================================

import streamlit as st
import pandas as pd
import requests
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

tqdm.pandas()

# ==============================================================================
# 🔹 Filter Data Terkait Polri
# ==============================================================================
def filter_polri(df, text_column='case_folded_text'):
    """Memfilter data agar hanya yang relevan dengan Polri."""
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
        return df

    mask_polri = df[text_column].str.contains(pattern_polri, flags=re.IGNORECASE, na=False)
    mask_exclude = df[text_column].str.contains(pattern_exclude, flags=re.IGNORECASE, na=False)

    df_filtered = df[mask_polri & ~mask_exclude].copy()
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
# 🔹 Fungsi Labeling 2 Kelas
# ==============================================================================
def label_sentiment_two_class(text, positive_lexicon, negative_lexicon):
    """Labeling hanya 2 kelas: positif dan negatif."""
    if not isinstance(text, str):
        return 'negatif'

    tokens = text.split()
    pos = sum(1 for t in tokens if t in positive_lexicon)
    neg = sum(1 for t in tokens if t in negative_lexicon)

    if pos == 0 and neg == 0:
        return 'negatif'

    return 'positif' if pos >= neg else 'negatif'

# ==============================================================================
# 🔹 Fungsi Preprocessing + Filter + Labeling
# ==============================================================================
@st.cache_data
def preprocess_filter_label(df, text_col, positive_lexicon, negative_lexicon):
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df.dropna(subset=['cleaned_text'], inplace=True)
    df['case_folded_text'] = df['cleaned_text'].str.lower()

    # Filter Polri
    df = filter_polri(df, text_column='case_folded_text')
    if df.empty:
        return df

    df['sentiment'] = df['case_folded_text'].progress_apply(
        lambda x: label_sentiment_two_class(x, positive_lexicon, negative_lexicon)
    )

    return df[['cleaned_text', 'case_folded_text', 'sentiment']]

# ==============================================================================
# 🔹 TF-IDF + Model NB & SVM
# ==============================================================================
def train_models(df, max_features=7000, test_size=0.3):
    X = df['case_folded_text']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    nb = MultinomialNB(alpha=0.2)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)

    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)

    results = {
        'nb': {'acc': nb_acc, 'report': classification_report(y_test, nb_pred, output_dict=True)},
        'svm': {'acc': svm_acc, 'report': classification_report(y_test, svm_pred, output_dict=True)},
        'vectorizer': vectorizer,
        'nb_model': nb,
        'svm_model': svm
    }

    return results

# ==============================================================================
# 🔹 Analisis Teks Tunggal
# ==============================================================================
def analyze_single_text(text, positive_lexicon, negative_lexicon):
    """Analisis cepat untuk input teks tunggal."""
    text_clean = clean_text(text.lower())
    filtered = pd.DataFrame({'case_folded_text': [text_clean]})
    filtered = filter_polri(filtered, 'case_folded_text')
    if filtered.empty:
        return "tidak relevan", text_clean
    sentiment = label_sentiment_two_class(text_clean, positive_lexicon, negative_lexicon)
    return sentiment, text_clean

# ==============================================================================
# 🔹 Load Lexicon InSet
# ==============================================================================
@st.cache_resource
def load_inset_lexicons():
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    pos = set(pd.read_csv(pos_url, sep='\t', header=None)[0].dropna().astype(str))
    neg = set(pd.read_csv(neg_url, sep='\t', header=None)[0].dropna().astype(str))
    return pos, neg

# ==============================================================================
# 🔹 UI STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Analisis Sentimen Polri", layout="wide")
st.title("📊 Analisis Sentimen Polri (Lexicon + Machine Learning)")
st.markdown("""
Aplikasi ini memiliki dua fitur:
1️⃣ **Upload file CSV** untuk analisis sentimen masal  
2️⃣ **Input teks langsung** untuk analisis cepat satu kalimat/paragraf  
""")

pos_lex, neg_lex = load_inset_lexicons()

# Tabs untuk dua mode
tab1, tab2 = st.tabs(["📂 Analisis File CSV", "⌨️ Analisis Teks Input"])

# ==============================================================================
# 🟦 TAB 1: UPLOAD FILE
# ==============================================================================
with tab1:
    uploaded_file = st.file_uploader("Unggah file CSV yang berisi teks", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"File berhasil diunggah: {uploaded_file.name} ({len(df)} baris)")

        text_col = st.selectbox("📝 Pilih kolom teks:", df.columns.tolist())

        if st.button("🚀 Jalankan Analisis File"):
            with st.spinner("Memproses data..."):
                df_processed = preprocess_filter_label(df, text_col, pos_lex, neg_lex)

            if not df_processed.empty:
                st.dataframe(df_processed.head(10))
                st.subheader("Distribusi Sentimen (2 Kelas)")
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
                    file_name="hasil_sentimen_polri_2kelas.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Tidak ada data relevan dengan Polri setelah filter.")

    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")

# ==============================================================================
# 🟩 TAB 2: INPUT TEKS
# ==============================================================================
with tab2:
    st.subheader("Analisis Cepat Teks Tunggal")
    input_text = st.text_area("Ketik atau paste teks Anda di sini:", height=150)

    if st.button("🔍 Analisis Teks Ini"):
        if input_text.strip():
            with st.spinner("Menganalisis teks..."):
                sentiment, cleaned = analyze_single_text(input_text, pos_lex, neg_lex)
            st.write("**Teks Setelah Preprocessing:**")
            st.info(cleaned)
            st.write("**Hasil Sentimen:**")
            if sentiment == "positif":
                st.success("✅ Positif 😊")
            else sentiment == "negatif":
                st.error("❌ Negatif 😠")
        else:
            st.warning("Masukkan teks terlebih dahulu sebelum menganalisis.")
