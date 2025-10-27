# ==============================================================================
# Aplikasi Streamlit: Analisis Sentimen Umum (Lexicon + ML)
# Pipeline: Preprocessing → Labeling (2 kelas) → TF-IDF → NB & SVM
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
from sklearn.svm import SVC # Menggunakan SVC kernel linear
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io # Untuk memuat lexicon dari request

tqdm.pandas()

# ==============================================================================
# 🔹 Fungsi Preprocessing Dasar
# ==============================================================================
def clean_text(text):
    """Membersihkan teks: hapus URL, mention, hashtag (#), karakter non-alfabet."""
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text) # Hapus URL
    text = re.sub(r'@\w+', '', text) # Hapus mention
    text = re.sub(r'#', ' ', text) # Ganti hashtag jadi spasi (simpan kata)
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hanya alfabet dan spasi
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

# ==============================================================================
# 🔹 Fungsi Labeling 2 Kelas (Positif/Negatif)
# ==============================================================================
def label_sentiment_two_class(text, positive_lexicon, negative_lexicon):
    """
    Memberi label Positif/Negatif berdasarkan jumlah kata.
    Default ke negatif jika skor sama atau tidak ada kata di lexicon.
    """
    if not isinstance(text, str) or not text.strip():
        return 'negatif' # Default jika teks kosong

    tokens = text.split()
    pos_count = sum(1 for t in tokens if t in positive_lexicon)
    neg_count = sum(1 for t in tokens if t in negative_lexicon)

    # Logika: Jika positif > negatif -> positif, selain itu negatif
    if pos_count > neg_count:
        return 'positif'
    else: # Termasuk jika pos_count == neg_count atau keduanya 0
        return 'negatif'

# ==============================================================================
# 🔹 Fungsi Preprocessing + Labeling (TANPA FILTER POLRI)
# ==============================================================================
@st.cache_data(show_spinner=False) # Spinner dikontrol manual
def preprocess_and_label(_df, text_col, positive_lexicon, negative_lexicon):
    """Menerapkan cleaning, case folding, dan labeling lexicon 2 kelas."""
    st.info("Memulai preprocessing & pelabelan...")
    df_processed = _df.copy() # Gunakan _df agar tidak konflik
    progress_bar = st.progress(0, text="Memulai...")

    # Langkah 1: Cleaning
    progress_bar.progress(1/3, text="1/3 Cleaning teks...")
    df_processed['cleaned_text'] = df_processed[text_col].astype(str).progress_apply(clean_text)
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    if df_processed.empty:
        st.warning("Tidak ada teks valid setelah cleaning.")
        progress_bar.empty()
        return df_processed # Kembalikan DataFrame kosong

    # Langkah 2: Case Folding
    progress_bar.progress(2/3, text="2/3 Case Folding...")
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()

    # Langkah 3: Labeling (pada case_folded_text)
    progress_bar.progress(3/3, text="3/3 Pelabelan Sentimen (Lexicon)...")
    df_processed['sentiment'] = df_processed['case_folded_text'].progress_apply(
        lambda x: label_sentiment_two_class(x, positive_lexicon, negative_lexicon)
    )

    progress_bar.empty() # Hapus progress bar
    st.success("Preprocessing & Pelabelan Selesai.")
    # Pilih kolom utama untuk dikembalikan
    return df_processed[[text_col, 'cleaned_text', 'case_folded_text', 'sentiment']]

# ==============================================================================
# 🔹 TF-IDF + Model NB & SVM
# ==============================================================================
# @st.cache_data # Cache ini mungkin kurang efektif jika data sering berubah
def train_models(_df_processed, max_features=7000, test_size=0.3):
    """Melatih Naive Bayes dan SVM (Linear) menggunakan TF-IDF."""
    st.info("Memulai pelatihan model Machine Learning...")
    if _df_processed.empty or 'sentiment' not in _df_processed.columns:
         st.error("Data yang diproses kosong atau tidak memiliki kolom 'sentiment'.")
         return None
    if len(_df_processed['sentiment'].unique()) < 2:
         st.error("Hanya ditemukan 1 kelas sentimen. Tidak dapat melatih model.")
         return None

    X = _df_processed['case_folded_text'] # Input fitur adalah teks yg sudah case folding
    y = _df_processed['sentiment']
    labels = sorted(y.unique()) # Dapatkan label unik (harus ['negatif', 'positif'])

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        st.write(f"Data dibagi: {len(X_train)} train, {len(X_test)} test (Test Size = {test_size:.0%})")
        st.write("Distribusi kelas (Train):", y_train.value_counts())
    except ValueError as e:
        st.error(f"Gagal membagi data (mungkin data terlalu sedikit atau hanya 1 kelas): {e}")
        return None

    # TF-IDF
    st.write(f"Membuat fitur TF-IDF (max_features={max_features}, ngram=1-2)...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    try:
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
    except Exception as e:
        st.error(f"Gagal saat TF-IDF: {e}")
        return None

    results = {}

    # Naive Bayes
    st.write("Melatih Naive Bayes...")
    nb = MultinomialNB(alpha=0.2) # Alpha dari notebook sebelumnya
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_report = classification_report(y_test, nb_pred, labels=labels, output_dict=True, zero_division=0)
    results['nb'] = {'acc': nb_acc, 'report': nb_report, 'model': nb, 'labels': labels}
    st.write(f"Akurasi Naive Bayes: {nb_acc*100:.2f}%")

    # SVM (Linear Kernel)
    st.write("Melatih SVM (Linear)...")
    svm = SVC(kernel='linear', random_state=42, probability=True) # probability=True jika butuh proba nanti
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred, labels=labels, output_dict=True, zero_division=0)
    results['svm'] = {'acc': svm_acc, 'report': svm_report, 'model': svm, 'labels': labels}
    st.write(f"Akurasi SVM: {svm_acc*100:.2f}%")

    # Simpan vectorizer juga
    results['vectorizer'] = vectorizer

    st.success("Pelatihan model ML selesai.")
    return results

# ==============================================================================
# 🔹 Analisis Teks Tunggal (TANPA FILTER POLRI)
# ==============================================================================
def analyze_single_text(text, positive_lexicon, negative_lexicon):
    """Analisis cepat untuk input teks tunggal menggunakan lexicon."""
    if not text or not text.strip():
        return "tidak valid", ""
    text_clean = clean_text(text) # Clean saja, tidak perlu case folding di sini
    if not text_clean:
         return "tidak valid", "" # Jika teks jadi kosong setelah cleaning
    text_case_folded = text_clean.lower() # Case folding untuk labeling
    # --- Filter Polri DIHAPUS ---
    sentiment = label_sentiment_two_class(text_case_folded, positive_lexicon, negative_lexicon)
    return sentiment, text_clean # Kembalikan teks cleaned (bukan case folded)

# ==============================================================================
# 🔹 Load Lexicon InSet (Tetap Sama)
# ==============================================================================
@st.cache_resource
def load_inset_lexicons():
    st.info("Memuat lexicon InSet...")
    pos_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv'
    neg_url = 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv'
    try:
        pos_df = pd.read_csv(pos_url, sep='\t', header=None, names=['word', 'weight'], on_bad_lines='skip', encoding='utf-8')
        neg_df = pd.read_csv(neg_url, sep='\t', header=None, names=['word', 'weight'], on_bad_lines='skip', encoding='utf-8')
        pos = set(pos_df['word'].dropna().astype(str))
        neg = set(neg_df['word'].dropna().astype(str))
        st.success(f"Ok: Lexicon InSet dimuat (Pos: {len(pos)}, Neg: {len(neg)}).")
        return pos, neg
    except Exception as e:
        st.error(f"Gagal memuat lexicon: {e}")
        return set(), set() # Kembalikan set kosong jika gagal

# ==============================================================================
# 🔹 UI STREAMLIT (Judul dan Deskripsi Disesuaikan)
# ==============================================================================
st.set_page_config(page_title="Analisis Sentimen Umum", layout="wide")
st.title("📊 Analisis Sentimen Umum (Lexicon + Machine Learning)") # Judul diubah
st.markdown("""
Aplikasi ini melakukan analisis sentimen pada teks berbahasa Indonesia menggunakan dua pendekatan:
1️⃣ **Pelabelan berbasis Leksikon**: Menggunakan kamus kata InSet untuk menentukan sentimen positif/negatif secara langsung.
2️⃣ **Klasifikasi Machine Learning**: Melatih model Naive Bayes dan SVM (Linear) menggunakan fitur TF-IDF dari teks yang telah dilabeli oleh leksikon.

Pilih mode **Analisis File CSV** atau **Analisis Teks Input**.
""")

# Muat lexicon sekali saat aplikasi dimulai
pos_lex, neg_lex = load_inset_lexicons()

# Tabs untuk dua mode
tab1, tab2 = st.tabs(["📂 Analisis File CSV", "⌨️ Analisis Teks Input"])

# ==============================================================================
# 🟦 TAB 1: UPLOAD FILE
# ==============================================================================
with tab1:
    st.header("Analisis Sentimen dari File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=['csv'], label_visibility="collapsed")

    # Inisialisasi state
    if 'processed_df' not in st.session_state: st.session_state.processed_df = None
    if 'ml_results' not in st.session_state: st.session_state.ml_results = None
    if 'current_filename' not in st.session_state: st.session_state.current_filename = ""

    if uploaded_file:
        # Reset state jika file baru
        if uploaded_file.name != st.session_state.current_filename:
            st.session_state.processed_df = None
            st.session_state.ml_results = None
            st.session_state.current_filename = uploaded_file.name
            if 'selectbox_column' in st.session_state: del st.session_state['selectbox_column']

        try:
            # Baca file dengan spinner
            with st.spinner(f"Membaca file {uploaded_file.name}..."):
                try: df_input = pd.read_csv(uploaded_file)
                except UnicodeDecodeError: uploaded_file.seek(0); df_input = pd.read_csv(uploaded_file, encoding='latin1')
            st.success(f"File berhasil diunggah: {uploaded_file.name} ({len(df_input)} baris)")
            st.dataframe(df_input.head(), hide_index=True)

            # Pilih kolom teks
            st.subheader("1. Pilih Kolom Teks")
            available_columns = [""] + df_input.columns.tolist()
            text_col = st.selectbox("Pilih kolom yang berisi teks:", available_columns, key="selectbox_column")

            if text_col:
                st.info(f"Kolom teks yang dipilih: **{text_col}**")

                # Tombol Proses Preprocessing & Labeling
                st.subheader("2. Preprocessing & Pelabelan Lexicon")
                if st.button("🔬 Proses Teks & Label", key="button_process_label"):
                    st.session_state.processed_df = None # Reset hasil sebelumnya
                    st.session_state.ml_results = None
                    # Validasi input sebelum proses
                    if text_col not in df_input.columns: st.error("Kolom teks tidak valid.")
                    elif df_input[text_col].isnull().all(): st.error(f"Kolom '{text_col}' tidak berisi data teks.")
                    else:
                        # Panggil fungsi preprocess_and_label (tanpa filter)
                        df_processed = preprocess_and_label(df_input, text_col, pos_lex, neg_lex)
                        st.session_state.processed_df = df_processed # Simpan hasil

                # Tampilkan hasil preprocessing & labeling jika ada
                if st.session_state.processed_df is not None:
                    if not st.session_state.processed_df.empty:
                        st.dataframe(st.session_state.processed_df.head(10), hide_index=True)
                        st.subheader("Distribusi Sentimen Hasil Labeling Lexicon (2 Kelas)")
                        st.bar_chart(st.session_state.processed_df['sentiment'].value_counts())

                        # Tombol Latih Model ML (muncul setelah labeling selesai)
                        st.subheader("3. Latih Model Machine Learning")
                        test_size_ml = st.slider("Pilih Ukuran Data Uji (Test Size):", 0.1, 0.5, 0.3, 0.05, format="%.0f%%", key="slider_test_ml")
                        max_features_ml = st.number_input("Jumlah Fitur TF-IDF Maksimal:", min_value=100, max_value=20000, value=7000, step=100, key="numinput_maxfeat")

                        if st.button("🤖 Latih Model NB & SVM", key="button_train_ml"):
                            st.session_state.ml_results = None # Reset hasil ML
                            with st.spinner("🔢 Melatih model ML..."):
                                results = train_models(st.session_state.processed_df, max_features=max_features_ml, test_size=test_size_ml)
                                st.session_state.ml_results = results # Simpan hasil ML

                    else:
                        st.warning("Tidak ada data valid ditemukan setelah preprocessing.")

                # Tampilkan Hasil ML jika ada
                if st.session_state.ml_results is not None:
                    st.subheader("Hasil Pelatihan Model")
                    results = st.session_state.ml_results
                    col1, col2 = st.columns(2)
                    if 'nb' in results: col1.metric("Akurasi Naive Bayes", f"{results['nb']['acc']:.2%}")
                    if 'svm' in results: col2.metric("Akurasi SVM (Linear)", f"{results['svm']['acc']:.2%}")

                    with st.expander("Lihat Laporan Klasifikasi Detail"):
                        if 'nb' in results: st.text("Naive Bayes:"); st.dataframe(pd.DataFrame(results['nb']['report']).transpose())
                        if 'svm' in results: st.text("SVM (Linear):"); st.dataframe(pd.DataFrame(results['svm']['report']).transpose())

                    # Tombol Unduh Hasil
                    st.subheader("4. Unduh Hasil Labeling")
                    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
                        csv_data = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
                        # Nama file download disesuaikan
                        base_filename = st.session_state.current_filename.split('.')[0] if '.' in st.session_state.current_filename else st.session_state.current_filename
                        download_filename = f"hasil_sentimen_{base_filename}_2kelas.csv"
                        st.download_button(
                            label="📥 Unduh CSV Hasil Labeling",
                            data=csv_data,
                            file_name=download_filename,
                            mime="text/csv",
                            key="download_csv"
                        )
                    else:
                        st.warning("Tidak ada data hasil labeling untuk diunduh.")

            elif not text_col:
                st.warning("☝️ Pilih kolom teks terlebih dahulu.")

        except pd.errors.EmptyDataError:
            st.error("File CSV yang diunggah kosong.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")

# ==============================================================================
# 🟩 TAB 2: INPUT TEKS (TANPA FILTER POLRI)
# ==============================================================================
with tab2:
    st.header("Analisis Cepat Teks Tunggal")
    input_text = st.text_area("Ketik atau paste teks Anda di sini:", height=150, key="text_area_single")

    if st.button("🔍 Analisis Teks Ini", key="button_analyze_single"):
        if input_text and input_text.strip():
            with st.spinner("Menganalisis teks..."):
                # Panggil analyze_single_text (yang sudah tidak ada filter)
                sentiment, cleaned = analyze_single_text(input_text, pos_lex, neg_lex)

            st.subheader("Hasil Analisis:")
            st.write("**Teks Setelah Preprocessing:**")
            st.info(f"`{cleaned}`") # Tampilkan teks cleaned
            st.write("**Hasil Sentimen (Lexicon):**")

            if sentiment == "positif":
                st.success("✅ Positif 😊")
            elif sentiment == "negatif":
                st.error("❌ Negatif 😠")
            elif sentiment == "tidak valid":
                 st.warning("⚠️ Teks tidak valid atau menjadi kosong setelah preprocessing.")
            # --- Kondisi "tidak relevan" dihapus karena filter dihilangkan ---

        else:
            st.warning("Masukkan teks terlebih dahulu sebelum menganalisis.")

# --- Footer ---
st.markdown("---")
st.markdown("Aplikasi Analisis Sentimen Dasar")
