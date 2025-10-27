# ==============================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML)
# Pipeline: Preprocessing (termasuk filter Polri) → Labeling (2 kelas) → TF-IDF → NB & SVM
# Dua Fitur: Upload File & Ketik Teks Input
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

tqdm.pandas()

# ==============================================================================
# 🔹 Preprocessing (Cleaning + Filter Polri)
# ==============================================================================
def preprocess_text(text):
    """Membersihkan teks dari URL, mention, hashtag, dan simbol."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def is_relevant_to_polri(text):
    """Cek apakah teks berkaitan dengan Polri dan bukan TNI/dll."""
    keywords_polri = [
        'polri', 'polisi', 'kapolri', 'brimob', 'polda',
        'polsek', 'polres', 'satlantas', 'bhayangkara', 'penyidik'
    ]
    exclude_keywords = [
        'tni', 'tentara', 'prajurit', 'kkb', 'ad', 'au', 'al', 'kostrad', 'kopassus'
    ]
    # Pola regex
    pattern_polri = r'\b(?:' + '|'.join(keywords_polri) + r')\b'
    pattern_exclude = r'\b(?:' + '|'.join(exclude_keywords) + r')\b'

    # Relevan jika mengandung kata Polri dan tidak mengandung TNI/dll
    return bool(re.search(pattern_polri, text)) and not re.search(pattern_exclude, text)

# ==============================================================================
# 🔹 Labeling (2 kelas: positif / negatif)
# ==============================================================================
def label_sentiment_two_class(text, positive_lexicon, negative_lexicon):
    """Labeling 2 kelas berdasarkan jumlah kata positif dan negatif."""
    if not isinstance(text, str):
        return 'negatif'

    tokens = text.split()
    pos = sum(1 for t in tokens if t in positive_lexicon)
    neg = sum(1 for t in tokens if t in negative_lexicon)

    if pos == 0 and neg == 0:
        return 'negatif'

    return 'positif' if pos >= neg else 'negatif'

# ==============================================================================
# 🔹 Preprocessing + Filter + Labeling (gabung)
# ==============================================================================
@st.cache_data
def preprocess_and_label(df, text_col, pos_lex, neg_lex):
    """Preprocessing lengkap: cleaning, filter Polri, labeling 2 kelas."""
    st.info("⚙️ Memulai preprocessing dan filter Polri...")

    df = df.copy()
    df[text_col] = df[text_col].astype(str)
    df['cleaned_text'] = df[text_col].apply(preprocess_text)

    # Filter hanya teks relevan dengan Polri
    df = df[df['cleaned_text'].apply(is_relevant_to_polri)]

    if df.empty:
        st.warning("⚠️ Tidak ada data relevan dengan Polri setelah filter.")
        return df

    # Labeling
    st.write("🏷️ Melakukan labeling (positif / negatif)...")
    df['sentiment'] = df['cleaned_text'].progress_apply(
        lambda x: label_sentiment_two_class(x, pos_lex, neg_lex)
    )

    st.success(f"✅ Labeling selesai: {df['sentiment'].value_counts().to_dict()}")
    return df[['cleaned_text', 'sentiment']]

# ==============================================================================
# 🔹 TF-IDF + Model Naive Bayes & SVM
# ==============================================================================
def train_models(df, max_features=7000, test_size=0.3):
    X = df['cleaned_text']
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
        'nb_acc': nb_acc,
        'svm_acc': svm_acc,
        'nb_report': classification_report(y_test, nb_pred, output_dict=True),
        'svm_report': classification_report(y_test, svm_pred, output_dict=True)
    }

    return results

# ==============================================================================
# 🔹 Analisis teks tunggal
# ==============================================================================
def analyze_single_text(text, pos_lex, neg_lex):
    cleaned = preprocess_text(text)
    if not is_relevant_to_polri(cleaned):
        return "tidak relevan", cleaned
    sentiment = label_sentiment_two_class(cleaned, pos_lex, neg_lex)
    return sentiment, cleaned

# ==============================================================================
# 🔹 Load Leksikon InSet
# ==============================================================================
@st.cache_resource
def load_inset_lexicons():
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    pos = set(pd.read_csv(pos_url, sep='\t', header=None)[0].dropna().astype(str))
    neg = set(pd.read_csv(neg_url, sep='\t', header=None)[0].dropna().astype(str))
    st.success(f"Leksikon dimuat: {len(pos)} positif, {len(neg)} negatif")
    return pos, neg

# ==============================================================================
# 🔹 UI STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Analisis Sentimen Polri", layout="wide")
st.title("📊 Analisis Sentimen Polri (Lexicon + Machine Learning)")
st.markdown("""
Pipeline: **Preprocessing (dengan filter Polri)** → Labeling (2 kelas) → TF-IDF → NB & SVM

Aplikasi ini memiliki dua mode:
1️⃣ Upload File CSV  
2️⃣ Input Teks Langsung
""")

pos_lex, neg_lex = load_inset_lexicons()

tab1, tab2 = st.tabs(["📂 Analisis File CSV", "⌨️ Analisis Teks Input"])

# ==============================================================================
# 🟦 TAB 1 — UPLOAD FILE
# ==============================================================================
with tab1:
    uploaded_file = st.file_uploader("Unggah file CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"File diunggah: {uploaded_file.name} ({len(df)} baris)")

        text_col = st.selectbox("📝 Pilih kolom teks:", df.columns.tolist())

        if st.button("🚀 Jalankan Analisis File"):
            with st.spinner("Memproses data..."):
                df_processed = preprocess_and_label(df, text_col, pos_lex, neg_lex)

            if not df_processed.empty:
                st.dataframe(df_processed.head(10))
                st.bar_chart(df_processed['sentiment'].value_counts())

                with st.spinner("🔢 Melatih model ML..."):
                    results = train_models(df_processed)

                st.success(f"✅ Akurasi NB: {results['nb_acc']:.2%} | SVM: {results['svm_acc']:.2%}")

                csv_data = df_processed.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Unduh Hasil CSV", csv_data, "hasil_sentimen_polri.csv", "text/csv")

            else:
                st.warning("Tidak ada data relevan dengan Polri setelah filter.")
    else:
        st.info("Unggah file CSV untuk memulai.")

# ==============================================================================
# 🟩 TAB 2 — INPUT TEKS
# ==============================================================================
with tab2:
    st.subheader("Analisis Cepat Teks Tunggal")
    input_text = st.text_area("Ketik atau paste teks di sini:", height=150)

    if st.button("🔍 Analisis Teks Ini"):
        if input_text.strip():
            with st.spinner("Menganalisis teks..."):
                sentiment, cleaned = analyze_single_text(input_text, pos_lex, neg_lex)

            st.write("**Teks Setelah Preprocessing:**")
            st.info(cleaned)

            st.write("**Hasil Sentimen:**")
            if sentiment == "positif":
                st.success("✅ Positif 😊")
            elif sentiment == "negatif":
                st.error("❌ Negatif 😠")
            else:
                st.warning("⚠️ Tidak relevan dengan Polri.")
        else:
            st.warning("Masukkan teks terlebih dahulu sebelum menganalisis.")
