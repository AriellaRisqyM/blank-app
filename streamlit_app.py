# ==============================================================================
# Aplikasi Streamlit: Analisis Sentimen (Lexicon + ML dengan Stemming & Tuning)
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
import io
import time # Untuk mengukur waktu tuning

# ML Imports
from sklearn.model_selection import train_test_split, GridSearchCV # Import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC # Menggunakan LinearSVC
from sklearn.linear_model import LogisticRegression # Import Logistic Regression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# Import Pipeline dari scikit-learn (karena tidak pakai SMOTE)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize # <-- Pastikan baris ini ada dan benar
# WordCloud Import
from wordcloud import WordCloud

# Configure tqdm for Streamlit
tqdm.pandas()

# ==============================================================================
# Cache Resource Loading Functions
# ==============================================================================
# (Fungsi-fungsi load_kamus_normalisasi, load_combined_stopwords, get_stemmer,
# load_tsv_dict, load_manual_dict, load_set, load_all_sentiment_dictionaries
# tetap sama seperti sebelumnya, pastikan @st.cache_resource digunakan)
@st.cache_resource
def load_kamus_normalisasi(url):
    st.info(f"Mengunduh kamus normalisasi...")
    try:
        response = requests.get(url); response.raise_for_status(); kamus = response.json()
        st.success(f"Ok: {len(kamus)} kata normalisasi.")
        return kamus
    except Exception as e: st.error(f"Gagal: {e}"); return {}
@st.cache_resource
def load_combined_stopwords(url_kamus_online):
    st.info("Memuat stopwords..."); sw_sastrawi=set(); sw_kamus=set()
    try:
        factory=StopWordRemoverFactory()
        sw_sastrawi=set(factory.get_stop_words())
        st.success(f"Ok: {len(sw_sastrawi)} stopwords Sastrawi.")
    except Exception as e: st.warning(f"Sastrawi gagal: {e}.")
    try:
        response=requests.get(url_kamus_online); response.raise_for_status()
        sw_kamus=set(response.text.splitlines())
        st.success(f"Ok: {len(sw_kamus)} stopwords online.")
    except Exception as e: st.warning(f"Stopwords online gagal: {e}.")
    combined = sw_sastrawi.union(sw_kamus); st.success(f"Total stopwords: {len(combined)}."); return combined
@st.cache_resource
def get_stemmer():
    st.info("Inisialisasi stemmer..."); factory=StemmerFactory(); stemmer=factory.create_stemmer(); st.success("Stemmer siap."); return stemmer
@st.cache_resource
def load_tsv_dict(url, key_col, val_col, header=None, sep='\t'):
    try:
        df=pd.read_csv(url, sep=sep, header=header, names=[key_col, val_col], on_bad_lines='skip', engine='python', encoding='utf-8') # tambahkan encoding utf-8
        df.dropna(subset=[key_col, val_col], inplace=True) # Tambahkan dropna
        df[val_col]=df[val_col].astype(str).str.replace(r'[+]', '', regex=True)
        valid_rows=pd.to_numeric(df[val_col], errors='coerce').notna()
        df_valid=df[valid_rows].copy() # Buat copy untuk menghindari SettingWithCopyWarning
        # Konversi ke int setelah memastikan valid
        df_valid[val_col] = pd.to_numeric(df_valid[val_col], errors='coerce').astype(int)
        # Hapus duplikat sebelum membuat dictionary
        df_valid.drop_duplicates(subset=[key_col], keep='last', inplace=True)
        return dict(zip(df_valid[key_col], df_valid[val_col]))
    except Exception as e: st.warning(f"Gagal TSV {url.split('/')[-1]}: {e}"); return {}
@st.cache_resource
def load_manual_dict(url):
    dictionary={}; processed_keys = set() # Tambahkan processed_keys
    try:
        r=requests.get(url); r.raise_for_status(); lines=r.text.splitlines()
        for line in lines:
            line=line.strip();
            if not line or line.startswith(('word ', 'phrase ', 'emo ')): continue
            parts=line.rsplit(' ', 1);
            if len(parts)==2:
                key, val=parts[0].strip(), parts[1].strip() # Pastikan key dan val di-strip
                if not key: continue # Lewati jika key kosong
                try:
                    score=int(val.replace('+', ''))
                    if key not in processed_keys: # Cek duplikat
                         dictionary[key]=score
                         processed_keys.add(key)
                except ValueError: pass
        return dictionary
    except Exception as e: st.warning(f"Gagal manual {url.split('/')[-1]}: {e}"); return {}
@st.cache_resource
def load_set(url):
    try: r=requests.get(url); r.raise_for_status(); return set(line.strip() for line in r.text.splitlines() if line.strip())
    except Exception as e: st.warning(f"Gagal set {url.split('/')[-1]}: {e}"); return set()
@st.cache_resource
def load_all_sentiment_dictionaries(urls):
    st.info("Memuat semua kamus sentimen..."); senti_dict={}
    senti_dict.update(load_tsv_dict(urls['inset_fajri_pos'], 'word', 'score', header=None, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_fajri_neg'], 'word', 'score', header=None, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_onpilot_pos'], 'word', 'weight', header=0, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_onpilot_neg'], 'word', 'weight', header=0, sep='\t'))
    try: r=requests.get(urls['senti_json']); r.raise_for_status(); senti_dict.update(r.json()); st.info("Ok: SentiStrength JSON.")
    except Exception as e: st.warning(f"Gagal Senti JSON: {e}")
    senti_dict.update(load_manual_dict(urls['emoticon']));
    senti_dict.update(load_manual_dict(urls['booster'])) # Muat booster juga ke senti_dict
    idiom_dict=load_manual_dict(urls['idiom']); sorted_idioms=sorted(idiom_dict.items(), key=lambda x: len(x[0]), reverse=True)
    negating_words=load_set(urls['negating']); question_words=load_set(urls['question'])
    st.success(f"Ok: Kamus utama ({len(senti_dict)}), idiom ({len(idiom_dict)}), negasi ({len(negating_words)}), tanya ({len(question_words)}).")
    return senti_dict, sorted_idioms, negating_words, question_words

# ==============================================================================
# Preprocessing Functions
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text); text = re.sub(r'@\w+', '', text); text = re.sub(r'#', ' ', text)
    # --- Modifikasi: Hanya hapus karakter non-alfanumerik KECUALI spasi ---
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip(); return text

def normalize_tokens(tokens, kamus_normalisasi):
    if not isinstance(tokens, list): return []
    return [kamus_normalisasi.get(word, word) for word in tokens]

def remove_stopwords(tokens, stop_words):
    if not isinstance(tokens, list): return []
    # --- Modifikasi: Hapus kata dengan panjang <= 2 ---
    return [word for word in tokens if word not in stop_words and len(word) > 2]

# --- Fungsi Stemming (sudah ada di Colab) ---
def stem_tokens(tokens, _stemmer):
    if not isinstance(tokens, list): return []
    try:
        stemmed_list = [_stemmer.stem(token) for token in tokens]
        return [word for word in stemmed_list if word] # Hapus string kosong hasil stem
    except Exception as e:
        # Menampilkan pesan error stemming hanya sekali agar tidak terlalu verbose
        # print(f"Error dalam stemming pada token: {e}")
        return tokens # Kembalikan token asli jika error

# ==============================================================================
# "Canggih" Labeling Function (SUDAH DIPERBAIKI)
# ==============================================================================
def label_sentiment_canggih(text, senti_dict, sorted_idioms, negating_words, question_words):
    if not isinstance(text, str) or not text.strip(): return 'netral'
    original_text_lower = text.lower(); pos_score = 0; neg_score = 0
    words_for_q_check = original_text_lower.split()
    if any(q_word in words_for_q_check for q_word in question_words): return 'netral'
    text_after_idioms = original_text_lower; processed_idiom = False
    for idiom, score in sorted_idioms:
        idiom_with_spaces = f" {idiom} "; text_padded = f" {text_after_idioms} "
        if idiom_with_spaces in text_padded:
            if score > 0: pos_score += score
            else: neg_score += abs(score)
            text_after_idioms = text_after_idioms.replace(idiom, " ", 1); processed_idiom = True
    words = text_after_idioms.split(); is_negated = False; num_words = len(words)
    for i, word in enumerate(words):
        if not word: continue
        if word in negating_words: is_negated = True; continue
        score = senti_dict.get(word, 0)
        if score != 0:
            if is_negated: score *= -1
            if i > 0:
                 # Cek apakah kata sebelumnya adalah booster DARI senti_dict
                 prev_word = words[i-1]
                 booster_score = senti_dict.get(prev_word, 0) # Ambil skor kata sebelumnya
                 # Logika SentiStrength: booster adalah -1, -2, +1, +2 (atau range lain jika kamus berbeda)
                 if booster_score in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]: # Sesuaikan range jika perlu
                     if score > 0: score += booster_score
                     elif score < 0: score -= booster_score # Memperkuat negatif
            # --- Perbaikan sintaks if/else ---
            if score > 0:
                pos_score += score
            else:
                neg_score += abs(score)
            # --- Akhir perbaikan ---
            is_negated = False
    if pos_score == 0 and neg_score == 0: return 'netral'
    if pos_score > neg_score * 1.5: return 'positif'
    elif neg_score > pos_score * 1.5: return 'negatif'
    else: return 'netral'

# ==============================================================================
# Preprocessing + Stemming + Labeling Function (Cacheable)
# ==============================================================================
@st.cache_data(show_spinner=False) # Spinner dikontrol manual di UI
def full_preprocess_and_label_df(_df, text_column, _kamus_normalisasi, _stop_words, _stemmer, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    """Applies all preprocessing steps including stemming and lexicon labeling."""
    st.info("Memulai preprocessing lengkap & pelabelan lexicon...")
    df_processed = _df.copy()

    total_steps = 7 # Jumlah total langkah untuk progress bar
    progress_bar = st.progress(0, text="Memulai...")

    # Langkah 1: Cleaning & Case Folding
    progress_bar.progress(1/total_steps, text="1/7 Cleaning & Case Folding...")
    df_processed['cleaned_text'] = df_processed[text_column].astype(str).apply(clean_text) # Pastikan string
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    if df_processed.empty: st.warning("Tidak ada teks valid setelah cleaning."); return pd.DataFrame()
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()

    # Langkah 2: Tokenizing
    progress_bar.progress(2/total_steps, text="2/7 Tokenizing...")
    try: # Perlu download 'punkt' jika belum
        nltk.data.find('tokenizers/punkt')
        df_processed['tokenized_text'] = df_processed['case_folded_text'].progress_apply(word_tokenize)
    except LookupError:
        st.info("Mengunduh NLTK Punkt..."); nltk.download('punkt'); st.info("Selesai unduh.")
        df_processed['tokenized_text'] = df_processed['case_folded_text'].progress_apply(word_tokenize)

    # Langkah 3: Normalization
    progress_bar.progress(3/total_steps, text="3/7 Normalization...")
    df_processed['normalized_text'] = df_processed['tokenized_text'].progress_apply(lambda tokens: normalize_tokens(tokens, _kamus_normalisasi))

    # Langkah 4: Stopword Removal
    progress_bar.progress(4/total_steps, text="4/7 Stopword Removal...")
    df_processed['stopwords_removed'] = df_processed['normalized_text'].progress_apply(lambda tokens: remove_stopwords(tokens, _stop_words))

    # Langkah 5: Stemming
    progress_bar.progress(5/total_steps, text="5/7 Stemming...")
    df_processed['stemmed_tokens'] = df_processed['stopwords_removed'].progress_apply(lambda tokens: stem_tokens(tokens, _stemmer))

    # Langkah 6: Join Stemmed Tokens
    progress_bar.progress(6/total_steps, text="6/7 Menggabungkan Teks Stemmed...")
    df_processed['stemmed_joined_text'] = df_processed['stemmed_tokens'].progress_apply(lambda tokens: ' '.join(tokens) if isinstance(tokens, list) else '')

    # Langkah 7: Lexicon Labeling (on 'case_folded_text')
    progress_bar.progress(7/total_steps, text="7/7 Pelabelan Sentimen (Lexicon)...")
    df_processed['sentiment'] = df_processed['case_folded_text'].progress_apply(
        lambda text: label_sentiment_canggih(text, _senti_dict, _sorted_idioms, _negating_words, _question_words)
    )
    progress_bar.empty() # Hapus progress bar setelah selesai

    st.success("Preprocessing Lengkap & Pelabelan Lexicon Selesai.")
    cols_to_return = [text_column, 'case_folded_text', 'stemmed_joined_text', 'sentiment'] + [col for col in df_processed.columns if col not in [text_column, 'case_folded_text','stemmed_joined_text','sentiment', 'cleaned_text', 'tokenized_text', 'normalized_text', 'stopwords_removed', 'stemmed_tokens']]
    return df_processed[cols_to_return]

# ==============================================================================
# ML Modeling Function (Dengan Tuning, Tanpa SMOTE)
# ==============================================================================
def train_evaluate_tuned_models_no_smote(df_labeled, selected_test_size=0.2, text_col_for_tfidf='stemmed_joined_text', target_col='sentiment'):
    """Performs Train/Test split, TF-IDF, trains NB, tunes+trains SVM & LR, returns results."""
    st.info("Memulai persiapan data dan pelatihan model ML (dengan tuning, tanpa SMOTE)...")
    start_time_total = time.time() # Mulai timer total

    if df_labeled.empty or target_col not in df_labeled.columns or text_col_for_tfidf not in df_labeled.columns:
        st.error("Dataframe input kosong atau kolom yang dibutuhkan tidak ada.")
        return None

    df_labeled = df_labeled.dropna(subset=[text_col_for_tfidf, target_col])
    if len(df_labeled[target_col].unique()) < 2: st.error(f"Hanya ada {len(df_labeled[target_col].unique())} kelas unik. Minimal 2."); return None
    if len(df_labeled) < 50: st.warning(f"Jumlah data ({len(df_labeled)}) sangat sedikit.")

    X = df_labeled[text_col_for_tfidf]
    y = df_labeled[target_col]
    labels = sorted(y.unique())

    # Split Data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=selected_test_size, random_state=42, stratify=y)
        st.write(f"Data dibagi: {len(X_train)} train, {len(X_test)} test (Test Size = {selected_test_size:.0%})")
        st.write("Distribusi kelas (Train):", y_train.value_counts())
    except ValueError as e: st.error(f"Gagal membagi data: {e}"); return None

    results = {}

    # --- Naive Bayes (Baseline) ---
    st.write("--- Melatih Naive Bayes (Baseline)...")
    start_time_nb = time.time()
    pipeline_nb = Pipeline([('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))), ('nb', MultinomialNB())])
    pipeline_nb.fit(X_train, y_train)
    y_pred_nb = pipeline_nb.predict(X_test)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    nb_report = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)
    nb_cm = confusion_matrix(y_test, y_pred_nb, labels=labels) # Ambil label dari y
    results['naive_bayes'] = {'accuracy': nb_accuracy, 'report': nb_report, 'cm': nb_cm, 'labels': labels, 'model': pipeline_nb}
    st.write(f"Akurasi Naive Bayes: {nb_accuracy*100:.2f}% (Waktu: {time.time() - start_time_nb:.2f} detik)")

    # --- SVM (Tuning) ---
    st.write("--- Melakukan Tuning SVM (LinearSVC)...")
    start_time_svm = time.time()
    pipeline_svm_tuned = Pipeline([('tfidf', TfidfVectorizer()), ('svm', LinearSVC(random_state=42, max_iter=3000, dual=True))])
    param_grid_svm = {'tfidf__max_features': [5000, 7000, None], 'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__min_df': [1, 3], 'svm__C': [0.1, 1, 10]}
    grid_search_svm = GridSearchCV(pipeline_svm_tuned, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1, verbose=0) # verbose=0 untuk UI bersih
    grid_search_svm.fit(X_train, y_train)
    best_svm_model = grid_search_svm.best_estimator_
    y_pred_svm_tuned = best_svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm_tuned)
    svm_report = classification_report(y_test, y_pred_svm_tuned, output_dict=True, zero_division=0)
    svm_cm = confusion_matrix(y_test, y_pred_svm_tuned, labels=labels)
    results['svm_tuned'] = {'accuracy': svm_accuracy, 'report': svm_report, 'cm': svm_cm, 'labels': labels, 'model': best_svm_model, 'best_params': grid_search_svm.best_params_, 'cv_score': grid_search_svm.best_score_}
    st.write(f"Akurasi SVM (Tuned): {svm_accuracy*100:.2f}% (CV Score: {grid_search_svm.best_score_:.2%}, Waktu: {time.time() - start_time_svm:.2f} detik)")
    st.write(f"Parameter SVM terbaik: `{grid_search_svm.best_params_}`")

    # --- Logistic Regression (Tuning) ---
    st.write("--- Melakukan Tuning Logistic Regression...")
    start_time_lr = time.time()
    pipeline_lr_tuned = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))])
    param_grid_lr = {'tfidf__max_features': [5000, 7000, None], 'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__min_df': [1, 3], 'lr__C': [0.1, 1, 10, 100]}
    grid_search_lr = GridSearchCV(pipeline_lr_tuned, param_grid_lr, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search_lr.fit(X_train, y_train)
    best_lr_model = grid_search_lr.best_estimator_
    y_pred_lr_tuned = best_lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr_tuned)
    lr_report = classification_report(y_test, y_pred_lr_tuned, output_dict=True, zero_division=0)
    lr_cm = confusion_matrix(y_test, y_pred_lr_tuned, labels=labels)
    results['logistic_regression_tuned'] = {'accuracy': lr_accuracy, 'report': lr_report, 'cm': lr_cm, 'labels': labels, 'model': best_lr_model, 'best_params': grid_search_lr.best_params_, 'cv_score': grid_search_lr.best_score_}
    st.write(f"Akurasi Logistic Regression (Tuned): {lr_accuracy*100:.2f}% (CV Score: {grid_search_lr.best_score_:.2%}, Waktu: {time.time() - start_time_lr:.2f} detik)")
    st.write(f"Parameter LR terbaik: `{grid_search_lr.best_params_}`")

    st.success(f"Pelatihan dan evaluasi model ML selesai. (Total Waktu: {time.time() - start_time_total:.2f} detik)")
    return results

# ==============================================================================
# Single Text Processing Function (sama seperti sebelumnya)
# ==============================================================================
def process_single_text(input_text, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    if not input_text or not isinstance(input_text, str) or not input_text.strip():
        return None, "Input teks kosong atau tidak valid."
    cleaned = clean_text(input_text)
    if not cleaned: return None, "Teks menjadi kosong setelah cleaning."
    case_folded = cleaned.lower()
    sentiment = label_sentiment_canggih(case_folded, _senti_dict, _sorted_idioms, _negating_words, _question_words)
    return sentiment, case_folded

# ==============================================================================
# Helper Function for Download
# ==============================================================================
@st.cache_data
def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# Word Cloud Function (sama seperti sebelumnya, input 'case_folded_text')
# ==============================================================================
@st.cache_data(show_spinner=False)
def generate_wordcloud(_df, sentiment_label, _stop_words, text_col='case_folded_text'):
    st.info(f"Membuat word cloud untuk sentimen '{sentiment_label}'...")
    text_data = _df[_df['sentiment'] == sentiment_label][text_col]
    if text_data.empty: st.write(f"Tidak ada data '{sentiment_label}'."); return None
    all_text = " ".join(text for text in text_data.astype(str)) # Pastikan string
    if not all_text.strip(): st.write(f"Tidak ada teks valid '{sentiment_label}'."); return None
    try:
        colormap = 'Greens' if sentiment_label == 'positif' else ('Reds' if sentiment_label == 'negatif' else 'Greys')
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=_stop_words, colormap=colormap, min_font_size=10, prefer_horizontal=0.9).generate(all_text)
        return wordcloud.to_array()
    except Exception as e: st.warning(f"Gagal word cloud '{sentiment_label}': {e}"); return None


# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="Analisis Sentimen Lexicon+ML")

st.title("📊 Aplikasi Analisis Sentimen (Lexicon + ML)")
st.markdown("""
Aplikasi ini melakukan preprocessing (termasuk **Stemming**), pelabelan sentimen berbasis **Lexicon** (InSet & SentiStrength),
dan melatih model **Machine Learning** (Naive Bayes, SVM, Logistic Regression) menggunakan TF-IDF pada data yang diunggah.
Model SVM dan Logistic Regression dioptimalkan menggunakan **Hyperparameter Tuning**. **SMOTE tidak digunakan**.
""")
st.info("Catatan: Proses preprocessing dan tuning hyperparameter mungkin memakan waktu beberapa menit tergantung ukuran data.")

# --- Load Resources ---
with st.spinner("⏳ Memuat sumber daya (kamus, stemmer)..."):
    url_kamus_norm = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt'
    url_stopwords_online = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt'
    lexicon_urls = { 'inset_fajri_pos': 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', 'inset_fajri_neg': 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', 'inset_onpilot_pos': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv', 'inset_onpilot_neg': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv', 'senti_json': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt', 'booster': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/boosterwords_id.txt', 'emoticon': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/emoticon_id.txt', 'idiom': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/idioms_id.txt', 'negating': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/negatingword.txt', 'question': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/questionword.txt' }

    # Muat semua resource menggunakan fungsi cache
    kamus_normalisasi = load_kamus_normalisasi(url_kamus_norm)
    stop_words = load_combined_stopwords(url_stopwords_online)
    stemmer = get_stemmer()
    senti_dict, sorted_idioms, negating_words, question_words = load_all_sentiment_dictionaries(lexicon_urls)

st.success("✅ Sumber daya siap.")
st.markdown("---")

# --- Tabs ---
tab1, tab2 = st.tabs(["📤 Unggah File (Preprocessing + Lexicon + ML)", "⌨️ Input Teks Langsung (Hanya Lexicon)"])

# --- Tab 1: File Upload ---
with tab1:
    st.header("1. Unggah Data Anda (CSV/Excel)")
    uploaded_file = st.file_uploader("Pilih file", type=['csv', 'xlsx'], label_visibility="collapsed", key="file_uploader")

    # Initialize session state
    if 'processed_df' not in st.session_state: st.session_state['processed_df'] = None
    if 'ml_results' not in st.session_state: st.session_state['ml_results'] = None
    if 'current_filename' not in st.session_state: st.session_state['current_filename'] = ""

    if uploaded_file is not None:
        # Clear state if new file
        if uploaded_file.name != st.session_state.current_filename:
            st.session_state.processed_df = None
            st.session_state.ml_results = None
            st.session_state.current_filename = uploaded_file.name

        try:
            if uploaded_file.name.endswith('.csv'): df_input = pd.read_csv(uploaded_file)
            else: df_input = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success(f"File '{uploaded_file.name}' diunggah ({len(df_input)} baris).")
            st.dataframe(df_input.head(), hide_index=True)

            st.header("2. Pilih Kolom Teks")
            available_columns = [""] + df_input.columns.tolist()
            text_column = st.selectbox("Pilih kolom:", options=available_columns, index=0, key="selectbox_column")

            if text_column:
                st.info(f"Kolom teks yang dipilih: **{text_column}**")

                # --- Preprocessing & Lexicon Labeling Button ---
                st.header("3. Proses Preprocessing Lengkap & Labeling Lexicon")
                if st.button("🔬 Proses Teks (Stemming) & Label Lexicon", key="button_process_lexicon"):
                    st.session_state.processed_df = None; st.session_state.ml_results = None # Clear previous

                    if text_column not in df_input.columns: st.error("Kolom tidak valid.")
                    elif df_input[text_column].isnull().all(): st.error(f"Kolom '{text_column}' kosong.")
                    else:
                        if not pd.api.types.is_string_dtype(df_input[text_column]):
                            try: df_input[text_column] = df_input[text_column].astype(str); st.warning(f"Kolom dikonversi ke teks.")
                            except Exception as e: st.error(f"Gagal konversi: {e}"); st.stop()

                        df_valid = df_input.dropna(subset=[text_column])
                        df_valid = df_valid[df_valid[text_column].astype(str).str.strip().astype(bool)]

                        if df_valid.empty: st.warning("Tidak ada teks valid.")
                        else:
                            # --- Panggil fungsi preprocessing lengkap ---
                            df_processed_labeled = full_preprocess_and_label_df(
                                df_valid, text_column,
                                kamus_normalisasi, stop_words, stemmer, # Preprocessing resources
                                senti_dict, sorted_idioms, negating_words, question_words # Labeling resources
                            )
                            if df_processed_labeled.empty: st.warning("Tidak ada hasil setelah proses.")
                            else: st.session_state.processed_df = df_processed_labeled # Store result

                # --- Display Preprocessing & Labeling Results ---
                if st.session_state.processed_df is not None:
                    st.success("🎉 Preprocessing Lengkap & Labeling Lexicon Selesai!")
                    st.subheader("Hasil (Contoh):")
                    # Tampilkan kolom teks asli, hasil stemming, dan label
                    st.dataframe(st.session_state.processed_df[[text_column, 'stemmed_joined_text', 'sentiment']].head(10), hide_index=True)
                    st.subheader("Distribusi Sentimen (Lexicon):")
                    sentiment_counts_lex = st.session_state.processed_df['sentiment'].value_counts()
                    st.bar_chart(sentiment_counts_lex)
                    st.write(sentiment_counts_lex)

                    # --- Word Clouds ---
                    st.subheader("Word Clouds Sentimen (Lexicon):")
                    st.write("Dibuat dari 'case_folded_text' (sebelum stemming agar lebih mudah dibaca).")
                    col_wc1, col_wc2 = st.columns(2)
                    with col_wc1:
                        st.markdown("<h5>Positif 😊</h5>", unsafe_allow_html=True)
                        wc_pos_array = generate_wordcloud(st.session_state.processed_df, 'positif', stop_words)
                        if wc_pos_array is not None: st.image(wc_pos_array, use_column_width=True)
                    with col_wc2:
                        st.markdown("<h5>Negatif 😠</h5>", unsafe_allow_html=True)
                        wc_neg_array = generate_wordcloud(st.session_state.processed_df, 'negatif', stop_words)
                        if wc_neg_array is not None: st.image(wc_neg_array, use_column_width=True)
                    st.markdown("---")

                    # --- ML Modeling Button ---
                    st.header("4. Latih & Evaluasi Model ML (Input: Teks Stemmed)")
                    st.warning("Model ML (NB, SVM Tuned, LR Tuned) akan dilatih menggunakan kolom 'sentiment' hasil lexicon sebagai target dan 'stemmed_joined_text' sebagai fitur.")

                    selected_test_size = st.slider(label="Pilih Test Size:", min_value=0.1, max_value=0.5, value=0.2, step=0.05, format="%.0f%%", key="slider_test_size")
                    st.info(f"Data uji: {selected_test_size:.0%}. Data Latih: {1-selected_test_size:.0%}")

                    if st.button("🤖 Latih & Tuning Model ML", key="button_train_ml"):
                        st.session_state.ml_results = None # Clear previous
                        with st.spinner("⏳ Melatih & mengevaluasi model ML (tuning mungkin lama)..."):
                            ml_results_dict = train_evaluate_tuned_models_no_smote(
                                st.session_state.processed_df,
                                selected_test_size=selected_test_size,
                                text_col_for_tfidf='stemmed_joined_text' # Gunakan teks stemmed
                            )
                            if ml_results_dict: st.session_state.ml_results = ml_results_dict

                    # --- Display ML Results ---
                    if st.session_state.ml_results is not None:
                        st.success("🎉 Pelatihan & Evaluasi Model ML Selesai!")
                        results_ml = st.session_state.ml_results

                        st.subheader("Hasil Evaluasi Model:")
                        # Tampilkan akurasi dalam kolom
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1: st.metric("Akurasi Naive Bayes", f"{results_ml['naive_bayes']['accuracy']:.2%}")
                        with col_m2: st.metric("Akurasi SVM (Tuned)", f"{results_ml['svm_tuned']['accuracy']:.2%}", help=f"CV Score: {results_ml['svm_tuned']['cv_score']:.2%}")
                        with col_m3: st.metric("Akurasi Log. Regression (Tuned)", f"{results_ml['logistic_regression_tuned']['accuracy']:.2%}", help=f"CV Score: {results_ml['logistic_regression_tuned']['cv_score']:.2%}")

                        # Tampilkan laporan klasifikasi dalam expander
                        with st.expander("Lihat Laporan Klasifikasi Detail"):
                            st.text("Naive Bayes:")
                            st.dataframe(pd.DataFrame(results_ml['naive_bayes']['report']).transpose())
                            st.text("SVM (Tuned):")
                            st.dataframe(pd.DataFrame(results_ml['svm_tuned']['report']).transpose())
                            st.text("Logistic Regression (Tuned):")
                            st.dataframe(pd.DataFrame(results_ml['logistic_regression_tuned']['report']).transpose())

                        # Tampilkan confusion matrix
                        st.subheader("Confusion Matrix:")
                        fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
                        models_to_plot = {
                            'Naive Bayes': ('naive_bayes', 'Greens', axes_cm[0]),
                            'SVM (Tuned)': ('svm_tuned', 'Blues', axes_cm[1]),
                            'Log. Regression (Tuned)': ('logistic_regression_tuned', 'Oranges', axes_cm[2])
                        }
                        for model_name, (key, cmap, ax) in models_to_plot.items():
                            if key in results_ml:
                                ConfusionMatrixDisplay(confusion_matrix=results_ml[key]['cm'], display_labels=results_ml[key]['labels']).plot(ax=ax, cmap=cmap, values_format='d')
                                ax.set_title(f"{model_name}\nAkurasi: {results_ml[key]['accuracy']:.2%}")
                                ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout(); st.pyplot(fig_cm)

                        # --- Download Button ---
                        st.header("5. Unduh Hasil Lengkap")
                        csv_data = convert_df_to_csv(st.session_state.processed_df)
                        output_filename = f"hasil_sentimen_lengkap_{st.session_state.current_filename.split('.')[0]}.csv"
                        st.download_button(label="📥 Unduh Data + Preprocessing + Label (.csv)", data=csv_data, file_name=output_filename, mime='text/csv', key="download_button")

                elif not text_column: st.warning("☝️ Pilih kolom teks untuk memulai proses.")

        except Exception as e: st.error(f"Error membaca atau memproses file: {e}")
    else: st.info("Unggah file CSV atau Excel di atas untuk memulai.")


# --- Tab 2: Text Input ---
with tab2:
    st.header("Analisis Teks Tunggal (Hanya Lexicon)")
    st.info("Fitur ini hanya menggunakan metode berbasis kamus (lexicon) pada teks sebelum stemming.")
    user_text = st.text_area("Ketik atau paste teks Anda:", height=150, key="text_area_input")

    if st.button("🚀 Analisis Teks Ini!", key="button_process_text"):
        if user_text and user_text.strip():
            with st.spinner("⏳ Menganalisis teks..."):
                hasil_sentimen, teks_diproses = process_single_text(
                    user_text, senti_dict, sorted_idioms, negating_words, question_words
                )
            st.subheader("Hasil Analisis Teks:")
            if hasil_sentimen:
                st.write("Teks setelah Cleaning & Case Folding:")
                st.info(f"`{teks_diproses}`")
                st.write("Hasil Sentimen (Lexicon):")
                if hasil_sentimen == 'positif': st.success(f"**{hasil_sentimen.upper()}** 😊")
                elif hasil_sentimen == 'negatif': st.error(f"**{hasil_sentimen.upper()}** 😠")
                else: st.warning(f"**{hasil_sentimen.upper()}** 😐")
            else: st.error(f"Gagal: {teks_diproses}")
        else: st.warning("☝️ Masukkan teks terlebih dahulu.")

# --- Footer ---
st.markdown("---")
st.markdown("Dibuat dengan Streamlit")
