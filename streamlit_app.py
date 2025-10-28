# =====================================================================
# STREAMLIT: Analisis Sentimen Polri (DISESUAIKAN DENGAN IPYNB)
# =====================================================================
import streamlit as st
import pandas as pd
import requests
import re
import json
import io 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm.auto import tqdm
import nltk # Diperlukan
from nltk.tokenize import word_tokenize # Diperlukan
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Diperlukan
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # Diperlukan
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC # Diubah dari SVC ke LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

# Inisialisasi tqdm untuk pandas
tqdm.pandas()

# Mengabaikan warning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen Polri (IPYNB)", layout="wide")
st.title("📊 Analisis Sentimen Polri (Logika IPYNB)")

# =====================================================================
# 0. FUNGSI CACHE (Untuk memuat model & kamus yang berat)
# =====================================================================

@st.cache_resource
def load_nltk_punkt():
    """Memastikan NLTK Punkt diunduh."""
    try:
        # Kita cek 'punkt_tab' secara spesifik
        nltk.data.find('tokenizers/punkt_tab') 
    except LookupError:
        st.info("Mengunduh NLTK 'punkt_tab' tokenizer...")
        nltk.download('punkt_tab') # Sesuai file IPYNB
        st.info("Unduhan 'punkt_tab' selesai.")


@st.cache_resource
def get_stemmer():
    """Memuat Sastrawi Stemmer (lambat)."""
    st.info("Memuat Sastrawi Stemmer (hanya sekali)...")
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def get_stopwords():
    """Memuat gabungan stopwords (Sastrawi + Online)."""
    st.info("Memuat kamus Stopwords (hanya sekali)...")
    stop_words_sastrawi = set()
    stop_words_kamus = set()
    try:
        factory = StopWordRemoverFactory()
        stop_words_sastrawi = set(factory.get_stop_words())
    except Exception as e:
        st.warning(f"Gagal memuat Sastrawi Stopwords: {e}")
    
    url_stopwords = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt'
    try:
        response = requests.get(url_stopwords)
        response.raise_for_status()
        stop_words_kamus = set(response.text.splitlines())
    except Exception as e:
        st.warning(f"Gagal memuat stopwords online: {e}")
        
    stop_words = stop_words_sastrawi.union(stop_words_kamus)
    st.info(f"Total {len(stop_words)} stopwords gabungan dimuat.")
    return stop_words

@st.cache_resource
def get_slang_dict():
    """Memuat kamus normalisasi (alay)."""
    st.info("Memuat kamus Normalisasi (hanya sekali)...")
    url_kamus = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt'
    kamus_normalisasi = {}
    try:
        response = requests.get(url_kamus)
        response.raise_for_status()
        kamus_normalisasi = response.json()
        st.info(f"Berhasil memuat {len(kamus_normalisasi)} kata dari kamus normalisasi.")
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal memuat kamus normalisasi dari URL: {e}")
    return kamus_normalisasi

# Muat semua resource saat aplikasi dimulai
load_nltk_punkt()
stemmer = get_stemmer()
stop_words = get_stopwords()
kamus_normalisasi = get_slang_dict()

# =====================================================================
# 1. PREPROCESSING & FILTER (Fungsi Sesuai IPYNB)
# =====================================================================

def research_safe_clean_text(text):
    """
    Cleaning versi IPYNB (mengganti karakter spesial dengan spasi).
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Ganti dengan spasi
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_relevant_to_polri(text_lower):
    """
    Mengecek relevansi teks (Keyword Lengkap Sesuai IPYNB).
    """
    keywords_polri = [
        "polri", "kepolisian", "mabes polri", "polda", "polres", "polsek", "polrestabes", "polresta",
        "brimob", "korbrimob", "gegana", "pelopor", "bareskrim", "ditreskrimum", "ditreskrimsus",
        "ditresnarkoba", "korlantas", "ditlantas", "satlantas", "intelkam", "satintelkam",
        "densus", "densus 88", "propam", "divpropam", "paminal", "wabprof", "provos",
        "polairud", "korpolairud", "sabhara", "samapta", "ditsamapta", "satsamapta",
        "binmas", "satbinmas", "bhabinkamtibmas", "polwan", "polisi", "kapolri", "wakapolri",
        "kapolda", "wakapolda", "kapolres", "wakapolres", "kapolsek", "wakapolsek", "penyidik",
        "reskrim", "kasat", "kanit", "jenderal polisi", "komjen", "irjen", "brigjen", "kombes",
        "akbp", "kompol", "akp", "iptu", "ipda", "aiptu", "aipda", "bripka", "brigpol",
        "brigadir", "briptu", "bripda", "bharada", "bharatu", "bharaka"
    ]
    exclude_keywords = [
        "tni", "tentara", "angkatandarat", "angkatanlaut", "angkatanudara", "tni ad", "tni al", "tni au",
        "kodam", "korem", "kodim", "koramil", "kostrad", "pangkostrad", "divif", "kopassus",
        "danjenkopassus", "marinir", "kormar", "pasmar", "kopaska", "denjaka", "paskhas",
        "korpaskhas", "denbravo", "armed", "kavaleri", "zeni", "arhanud", "yonif",
        "prajurit", "panglima tni", "ksad", "kasad", "ksal", "kasal", "ksau", "kasau",
        "pangdam", "danrem", "dandim", "danramil", "jenderal tni", "laksamana", "marsekal",
        "letjen", "laksdya", "marsdya", "mayjen", "laksda", "marsda", "brigjen tni", "laksma",
        "marsma", "kolonel", "letkol", "mayor", "kapten", "lettu", "letda", "peltu", "pelda",
        "serma", "serka", "sertu", "serda", "kopka", "koptu", "kopda", "praka", "pratu", "prada"
    ]

    pattern_polri = r"\b(?:{})\b".format("|".join(keywords_polri))
    pattern_exclude = r"\b(?:{})\b".format("|".join(exclude_keywords))
    return bool(re.search(pattern_polri, text_lower)) and not re.search(pattern_exclude, text_lower)

# =====================================================================
# 2. LOAD LEXICON POSITIF & NEGATIF (Hanya Fajri91)
# =====================================================================
@st.cache_resource
def load_lexicons():
    """Memuat leksikon InSet (HANYA fajri91)."""
    st.info("📚 Memuat kamus positif & negatif (fajri91)...")
    urls = {
        "fajri_pos": "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        "fajri_neg": "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv",
    }
    pos_lex = set()
    neg_lex = set()
    try:
        pos_lex.update(set(pd.read_csv(io.StringIO(requests.get(urls["fajri_pos"]).text), sep="\t", header=None, usecols=[0], names=['word'], on_bad_lines='skip', encoding='utf-8')['word'].dropna().astype(str)))
        neg_lex.update(set(pd.read_csv(io.StringIO(requests.get(urls["fajri_neg"]).text), sep="\t", header=None, usecols=[0], names=['word'], on_bad_lines='skip', encoding='utf-8')['word'].dropna().astype(str)))
        st.success(f"✅ Leksikon dimuat: {len(pos_lex)} pos, {len(neg_lex)} neg.")
    except Exception as e:
        st.error(f"⚠️ Gagal memuat leksikon fajri91: {e}")
    return pos_lex, neg_lex

# Muat leksikon saat aplikasi dimulai
pos_lex, neg_lex = load_lexicons()

# =====================================================================
# 3. LABEL SENTIMEN (Logika IPYNB: pos >= neg)
# =====================================================================
def label_sentiment_two_class(tokens, pos_lex, neg_lex):
    """
    Memberi label Positif/Negatif (input adalah list token).
    Logika IPYNB: Jika positif >= negatif -> positif.
    """
    if not isinstance(tokens, list):
        return 'negatif' 
    pos = sum(1 for t in tokens if t in pos_lex)
    neg = sum(1 for t in tokens if t in neg_lex)
    # Logika 2 Kelas IPYNB
    if pos >= neg: # Logika IPYNB
        return "positif"
    else:
        return "negatif"

# =====================================================================
# 4. FUNGSI HELPER PIPELINE (Sesuai IPYNB)
# =====================================================================

def normalize_tokens(tokens, kamus):
    if not isinstance(tokens, list): return []
    return [kamus.get(word, word) for word in tokens]

def remove_stopwords(tokens, stop):
    if not isinstance(tokens, list): return []
    return [word for word in tokens if word not in stop]

def research_safe_stemming(tokens, stemmer_obj):
    if not isinstance(tokens, list): return []
    if stemmer_obj is None: return tokens
    try:
        stemmed_list = [stemmer_obj.stem(token) for token in tokens]
        return [word for word in stemmed_list if word]
    except Exception as e:
        st.warning(f"Error saat stemming: {e}")
        return tokens

# =====================================================================
# 5. PREPROCESS + FILTER + LABEL (Fungsi Terpadu - Sesuai IPYNB)
# =====================================================================
@st.cache_data(show_spinner=False)
def preprocess_and_label(_df, text_col, _pos_lex, _neg_lex, _kamus_norm, _stop_words, _stemmer):
    """Menerapkan pipeline preprocessing LENGKAP dari IPYNB."""
    st.info("Memulai preprocessing, filter, & pelabelan (Alur IPYNB)...")
    df_processed = _df.copy()
    total_awal = len(df_processed)
    
    # *** BAGIAN PERSENTASE BERJALAN (PERBAIKAN) ***
    progress_bar = st.progress(0, text="Memulai...")
    total_steps = 9 # Diubah ke 9 (Clean, Dedupe, Fold, Filter, Token, Norm, Stopword, Stem, Label)
    current_step = 0

    def update_progress(text):
        nonlocal current_step
        current_step += 1
        progress_bar.progress(current_step / total_steps, text=f"{current_step}/{total_steps} {text}...")
    # **********************************

    # Langkah 1: Cleaning (research_safe_clean_text)
    update_progress("Cleaning (Research Safe)")
    df_processed['cleaned_text'] = df_processed[text_col].astype(str).progress_apply(research_safe_clean_text)
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    if df_processed.empty:
        st.warning("Tidak ada teks valid setelah cleaning.")
        progress_bar.empty()
        return pd.DataFrame(), total_awal, 0, 0, 0, 0, 0, 0, 0

    # Langkah 2: De-duplikasi 'cleaned_text'
    update_progress("De-duplikasi Teks")
    rows_before_dedup = len(df_processed)
    df_processed.drop_duplicates(subset=['cleaned_text'], inplace=True)
    rows_after_dedup = len(df_processed)
    st.write(f"De-duplikasi 'cleaned_text': {rows_before_dedup - rows_after_dedup} duplikat dihapus.")
    df_processed = df_processed.reset_index(drop=True)

    # Langkah 3: Case Folding
    update_progress("Case Folding")
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()

    # Langkah 4: Filter Polri (Keyword Lengkap)
    update_progress("Memfilter data Polri")
    mask_polri = df_processed["case_folded_text"].progress_apply(is_relevant_to_polri)
    df_filtered = df_processed[mask_polri].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    total_filtered = len(df_filtered)
    if df_filtered.empty:
        st.warning("Tidak ada data relevan dengan Polri setelah filter.")
        progress_bar.empty()
        return pd.DataFrame(), total_awal, rows_after_dedup, total_filtered, 0, 0, 0, 0, 0
    
    # Langkah 5: Tokenizing (NLTK)
    update_progress("Tokenizing (NLTK)")
    df_filtered['tokenized_text'] = df_filtered['case_folded_text'].progress_apply(word_tokenize)
    total_tokenized = len(df_filtered)

    # Langkah 6: Normalization (Kamus Alay)
    update_progress("Normalization (Kamus Alay)")
    df_filtered['normalized_text'] = df_filtered['tokenized_text'].progress_apply(lambda x: normalize_tokens(x, _kamus_norm))
    total_normalized = len(df_filtered)
    
    # Langkah 7: Stopword Removal
    update_progress("Stopword Removal")
    df_filtered['stopwords_removed'] = df_filtered['normalized_text'].progress_apply(lambda x: remove_stopwords(x, _stop_words))
    total_stopwords = len(df_filtered)
    
    # Langkah 8: Stemming
    update_progress("Stemming (Sastrawi)")
    df_filtered['stemmed_tokens'] = df_filtered['stopwords_removed'].progress_apply(lambda x: research_safe_stemming(x, _stemmer))
    total_stemmed = len(df_filtered)
    
    # Langkah 9: Labeling (Input: stemmed_tokens)
    update_progress("Pelabelan Sentimen (Lexicon)")
    df_filtered["sentiment"] = df_filtered["stemmed_tokens"].progress_apply( 
        lambda x: label_sentiment_two_class(x, _pos_lex, _neg_lex)
    )
    total_label = len(df_filtered)
    progress_bar.empty()

    st.success("Preprocessing, Filter, & Pelabelan Selesai (Alur IPYNB).")
    
    return df_filtered, total_awal, rows_after_dedup, total_filtered, total_tokenized, total_normalized, total_stopwords, total_stemmed, total_label

# =====================================================================
# 6. TRAIN MODEL + TF-IDF (DENGAN KONTROL UI)
# =====================================================================
@st.cache_data(show_spinner=False)
def train_models(_df_processed, max_features=5000, test_size=0.2): # Argumen ditambahkan
    """Melatih Naive Bayes dan LinearSVC menggunakan TF-IDF (Sesuai IPYNB)."""
    
    RANDOM_STATE = 42

    st.info(f"Memulai pelatihan model (Test Size: {test_size:.0%}, Max Features: {max_features or 'Semua'})...")
    
    if _df_processed.empty or 'sentiment' not in _df_processed.columns:
         st.error("Data yang diproses kosong atau tidak memiliki kolom 'sentiment'.")
         return None
    if len(_df_processed['sentiment'].unique()) < 2:
         st.error("Hanya ditemukan 1 kelas sentimen. Tidak dapat melatih model.")
         return None

    # Input TF-IDF adalah 'case_folded_text' (Sesuai IPYNB)
    X, y = _df_processed["case_folded_text"], _df_processed["sentiment"]
    labels = sorted(y.unique())

    try:
        # Menggunakan test_size dari UI
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
        )
        st.write(f"Data dibagi: {len(X_train)} train, {len(X_test)} test")
    except ValueError as e:
        st.error(f"Gagal membagi data (mungkin data terlalu sedikit): {e}")
        return None

    st.write(f"Membuat fitur TF-IDF (max_features={max_features or 'Semua'}, ngram=1-2)...")
    # Menggunakan max_features dari UI (bukan sublinear_tf=True)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)) 
    try:
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
    except Exception as e:
        st.error(f"Gagal saat TF-IDF: {e}"); return None

    results = {"labels": labels}

    st.write("Melatih Naive Bayes...")
    nb = MultinomialNB() # Parameter default sesuai IPYNB
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_report = classification_report(y_test, nb_pred, labels=labels, output_dict=True, zero_division=0)
    results['nb'] = {'acc': nb_acc, 'report': nb_report, 'model': nb, 'preds': nb_pred}
    st.write(f"Akurasi Naive Bayes: {nb_acc*100:.2f}%")

    st.write("Melatih SVM (LinearSVC)...")
    svm = LinearSVC(random_state=RANDOM_STATE) # Sesuai IPYNB
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred, labels=labels, output_dict=True, zero_division=0)
    results['svm'] = {'acc': svm_acc, 'report': svm_report, 'model': svm, 'preds': svm_pred}
    st.write(f"Akurasi SVM (LinearSVC): {svm_acc*100:.2f}%")

    results.update({
        "vectorizer": vectorizer, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test
    })

    st.success("Pelatihan model ML selesai.")
    return results

# =====================================================================
# 7. VISUALISASI
# =====================================================================
def show_confusion(y_test, preds, model_name, labels):
    """Menampilkan confusion matrix."""
    cm_labels = sorted(list(set(y_test) | set(preds)))
    effective_labels = labels if all(l in cm_labels for l in labels) else cm_labels
    
    cm = confusion_matrix(y_test, preds, labels=effective_labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=effective_labels, yticklabels=effective_labels, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)

def show_wordcloud(_df):
    """Menampilkan word cloud (input 'case_folded_text' sesuai IPYNB)."""
    st.header("🌈 Tahap 3: WordCloud Sentimen")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positif 😊")
        text_pos = " ".join(_df[_df["sentiment"] == "positif"]["case_folded_text"].values)
        if text_pos.strip():
            try:
                wc_pos = WordCloud(width=600, height=300, background_color="white", colormap="Greens").generate(text_pos)
                st.image(wc_pos.to_array(), use_container_width=True) 
            except Exception as e:
                st.warning(f"Gagal membuat word cloud positif: {e}")
        else:
            st.write("Tidak ada data positif untuk word cloud.")
    with col2:
        st.subheader("Negatif 😠")
        text_neg = " ".join(_df[_df["sentiment"] == "negatif"]["case_folded_text"].values)
        if text_neg.strip():
            try:
                wc_neg = WordCloud(width=600, height=300, background_color="white", colormap="Reds").generate(text_neg)
                st.image(wc_neg.to_array(), use_container_width=True) 
            except Exception as e:
                st.warning(f"Gagal membuat word cloud negatif: {e}")
        else:
            st.write("Tidak ada data negatif untuk word cloud.")

# =====================================================================
# 8. ANALISIS TEKS TUNGGAL (TAB 2 - Disesuaikan ke Alur IPYNB)
# =====================================================================
def analyze_single_text(text, positive_lexicon, negative_lexicon, _kamus_norm, _stop_words, _stemmer):
    """
    Analisis cepat teks tunggal (Alur Preprocessing LENGKAP IPYNB).
    """
    if not text or not text.strip():
        return "tidak valid", "" 

    # 1. Clean (Research Safe)
    text_clean = research_safe_clean_text(text)
    if not text_clean:
        return "tidak valid", "" 

    # 2. Lower
    text_lower = text_clean.lower()
    
    # 3. Tokenize (NLTK)
    tokens = word_tokenize(text_lower)
    
    # 4. Normalize
    tokens = normalize_tokens(tokens, _kamus_norm)
    
    # 5. Stopword Removal
    tokens = remove_stopwords(tokens, _stop_words)
    
    # 6. Stemming
    tokens = research_safe_stemming(tokens, _stemmer)

    # 7. Label (Logika IPYNB: >=)
    sentiment = label_sentiment_two_class(tokens, positive_lexicon, negative_lexicon)
    return sentiment, text_clean # Kembalikan teks cleaned asli

# =====================================================================
# 9. UI: FILE CSV & TEKS TUNGGAL
# =====================================================================
# State management
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'ml_results' not in st.session_state: st.session_state.ml_results = None
if 'current_filename' not in st.session_state: st.session_state.current_filename = ""

tab1, tab2 = st.tabs(["📂 Analisis File CSV (Filter Polri)", "⌨️ Analisis Teks Umum"])

with tab1:
    st.header("Analisis Sentimen dari File CSV (Filter Polri)")
    st.info("Tab ini akan memfilter data dan hanya menganalisis teks yang relevan dengan Polri (menggunakan alur IPYNB).")
    uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"], label_visibility="collapsed")
    
    if uploaded and uploaded.name != st.session_state.current_filename:
        st.session_state.processed_df = None
        st.session_state.ml_results = None
        st.session_state.current_filename = uploaded.name
        if 'selectbox_column' in st.session_state: del st.session_state['selectbox_column']

    if uploaded:
        try:
            with st.spinner(f"Membaca {uploaded.name}..."):
                try: df_input = pd.read_csv(uploaded)
                except UnicodeDecodeError: uploaded.seek(0); df_input = pd.read_csv(uploaded, encoding='latin1')
            
            # (Tambahan: dedupe raw 'full_text' seperti di IPYNB Sel 3)
            if 'full_text' in df_input.columns:
                 df_input.dropna(subset=['full_text'], inplace=True)
                 df_input.drop_duplicates(subset=['full_text'], inplace=True)
                 df_input = df_input.reset_index(drop=True)
            
            st.success(f"File berhasil diunggah: {uploaded.name} ({len(df_input)} baris setelah de-dup awal)")
            st.dataframe(df_input.head(), hide_index=True)

            available_columns = [""] + df_input.columns.tolist()
            col_index = 0
            if 'selectbox_column' in st.session_state and st.session_state.selectbox_column in available_columns:
                col_index = available_columns.index(st.session_state.selectbox_column)
            text_col = st.selectbox("Pilih Kolom Teks:", available_columns, index=col_index, key="selectbox_column_tab1")

            if text_col:
                st.info(f"Kolom teks yang dipilih: **{text_col}**")
                
                # *** (INI ADALAH BAGIAN PERSENTASE BERJALAN) ***
                if st.button("🚀 1. Jalankan Preprocessing, Filter & Labeling (Alur IPYNB)", key="btn_process"):
                    st.session_state.processed_df = None
                    st.session_state.ml_results = None
                    if text_col not in df_input.columns: st.error("Kolom teks tidak valid.")
                    elif df_input[text_col].isnull().all(): st.error(f"Kolom '{text_col}' kosong.")
                    else:
                        # Panggil fungsi preprocess LENGKAP
                        df_processed, total_awal, total_clean_dedup, total_filtered, \
                        total_tokenized, total_normalized, total_stopwords, \
                        total_stemmed, total_label = preprocess_and_label(
                            df_input, text_col, pos_lex, neg_lex, 
                            kamus_normalisasi, stop_words, stemmer
                        )
                        st.session_state.processed_df = df_processed
                        
                        st.header("🧩 Hasil Preprocessing, Filter & Labeling")
                        colA, colB, colC = st.columns(3)
                        colA.metric("Total Data Awal", total_awal)
                        colB.metric("Data Setelah Clean & Dedupe", total_clean_dedup)
                        colC.metric("Data Setelah Filter Polri", total_filtered)
                        st.write(f"Data ditokenisasi: {total_tokenized}, Dinormalisasi: {total_normalized}, Stopwords: {total_stopwords}, Stemmed: {total_stemmed}, Dilabeli: {total_label}")


                if st.session_state.processed_df is not None:
                    if not st.session_state.processed_df.empty:
                        st.dataframe(st.session_state.processed_df.head(10), hide_index=True)
                        st.bar_chart(st.session_state.processed_df["sentiment"].value_counts())
                        
                        show_wordcloud(st.session_state.processed_df)

                        st.header("🤖 2. Pelatihan Model Machine Learning")
                        st.markdown("Model akan dilatih menggunakan **Label Lexicon** sebagai target dan **Teks (Case Folded)** sebagai fitur (Sesuai IPYNB).")
                        
                        # *** INI ADALAH KONTROL FITUR & TEST SIZE YANG DITAMBAHKAN KEMBALI ***
                        col_ml1, col_ml2 = st.columns(2)
                        with col_ml1:
                            test_size_percent = st.slider(
                                "Pilih Test Size (Data Uji)", 
                                10, 50, 20, step=5, # Default 20% sesuai IPYNB
                                format="%d%%", 
                                key="slider_test_tab1_ipynb" # Key unik
                            )
                            test_size = test_size_percent / 100.0 

                        with col_ml2:
                            max_feat_input = st.number_input(
                                "Max Features TF-IDF (0 = Semua)", 
                                min_value=0, 
                                max_value=100000, 
                                value=5000, # Default 5000 sesuai IPYNB
                                step=100, 
                                key="numinput_maxfeat_tab1_ipynb", # Key unik
                                help="Masukkan jumlah fitur/kata. Masukkan 0 untuk menggunakan semua fitur."
                            )
                            max_feat = None if max_feat_input == 0 else int(max_feat_input)
                        
                        display_max_feat = "Semua" if max_feat is None else max_feat
                        st.info(f"Data uji: {test_size:.0%}. Data Latih: {1-test_size:.0%}. Max Features: {display_max_feat}")
                        # ********************************************************************

                        if st.button("🤖 Latih Model NB & LinearSVC", key="btn_train"):
                            st.session_state.ml_results = None
                            with st.spinner("🔢 Melatih model ML (Logika IPYNB)..."):
                                # Memanggil train_models dengan nilai dari UI
                                results = train_models(st.session_state.processed_df, max_feat, test_size) 
                                st.session_state.ml_results = results

                    else:
                        st.warning("⚠️ Tidak ada data relevan dengan Polri setelah filter.")

                if st.session_state.ml_results is not None:
                    results = st.session_state.ml_results
                    st.header("📈 3. Hasil & Evaluasi Model")
                    
                    colD, colE = st.columns(2)
                    colD.metric("Akurasi Naive Bayes", f"{results['nb']['acc']:.2%}")
                    colE.metric("Akurasi SVM (LinearSVC)", f"{results['svm']['acc']:.2%}") # Diubah

                    st.subheader("Confusion Matrix")
                    colF, colG = st.columns(2)
                    with colF:
                        show_confusion(results["y_test"], results["nb"]["preds"], "Naive Bayes", results["labels"])
                    with colG:
                        show_confusion(results["y_test"], results["svm"]["preds"], "SVM (LinearSVC)", results["labels"]) # Diubah
                    
                    st.subheader("Laporan Klasifikasi Detail")
                    with st.expander("Lihat Laporan Naive Bayes"):
                        st.dataframe(pd.DataFrame(results['nb']['report']).transpose())
                    with st.expander("Lihat Laporan SVM (LinearSVC)"): # Diubah
                        st.dataframe(pd.DataFrame(results['svm']['report']).transpose())

                    st.header("📥 4. Unduh Hasil Labeling")
                    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
                        # (IPYNB menyimpan semua kolom, jadi kita lakukan hal yang sama)
                        cols_to_save = ['cleaned_text', 'case_folded_text', 'tokenized_text', 'normalized_text', 'stopwords_removed', 'stemmed_tokens', 'sentiment']
                        cols_existing = [col for col in cols_to_save if col in st.session_state.processed_df.columns]
                        
                        csv_data = st.session_state.processed_df[cols_existing].to_csv(index=False).encode("utf-8")
                        base_filename = st.session_state.current_filename.split('.')[0] if '.' in st.session_state.current_filename else st.session_state.current_filename
                        st.download_button(
                            "Unduh CSV Hasil Labeling (Full Preprocessing)",
                            csv_data,
                            f"hasil_sentimen_polri_full_{base_filename}.csv",
                            "text/csv",
                            key="download_csv_final"
                        )
                
            elif not text_col:
                st.warning("☝️ Pilih kolom teks terlebih dahulu.")
                
        except pd.errors.EmptyDataError:
            st.error("File CSV yang diunggah kosong.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.exception(e)

    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")

# ==============================================================================
# 🟩 TAB 2: INPUT TEKS (Disesuaikan ke Alur IPYNB)
# ==============================================================================
with tab2:
    st.header("💬 Analisis Cepat Teks Tunggal (Umum - Alur IPYNB)")
    st.info("Fitur ini menganalisis sentimen teks apapun (tanpa filter Polri) menggunakan alur preprocessing LENGKAP (Termasuk Stemming & Normalisasi).")
    input_text = st.text_area("Ketik atau paste teks di sini:", height=150, key="text_area_single")

    if st.button("🔍 Analisis Teks Ini", key="button_analyze_single"):
        if input_text and input_text.strip():
            with st.spinner("Menganalisis teks (Alur IPYNB)..."):
                
                # Panggil fungsi analisis lengkap (Sesuai IPYNB)
                sentiment, cleaned_display = analyze_single_text(
                    input_text, pos_lex, neg_lex, 
                    kamus_normalisasi, stop_words, stemmer
                )
                        
            st.subheader("Hasil Analisis:")
            st.write("**Teks Setelah Preprocessing (Cleaned):**")
            st.info(f"`{cleaned_display}`") 
            st.write("**Hasil Sentimen:**")

            if sentiment == "positif":
                st.success("✅ Sentimen: POSITIF 😊")
            elif sentiment == "negatif":
                st.error("❌ Sentimen: NEGATIF 😠")
            else: # 'tidak valid'
               st.warning("⚠️ Teks tidak valid atau menjadi kosong setelah preprocessing.")

        else:
            st.warning("Masukkan teks terlebih dahulu sebelum menganalisis.")

# --- Footer ---
st.markdown("---")
st.markdown("Aplikasi Analisis Sentimen (Menggunakan Logika IPYNB)")
