# ==============================================================================
# Aplikasi Streamlit: Analisis Sentimen (Lexicon + ML dengan Stemming & Tuning)
# ==============================================================================
import streamlit as st
import pandas as pd
import requests
import json
import re
import nltk # <- Pastikan nltk diimpor
from nltk.tokenize import word_tokenize # <- Pastikan word_tokenize diimpor di sini
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm.auto import tqdm
import io
import time

# ML Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# WordCloud Import
from wordcloud import WordCloud

# Configure tqdm for Streamlit
tqdm.pandas()

# ==============================================================================
# STEP 1: SETUP AWAL (MEMUAT LIBRARY DAN DATASET)
# ==============================================================================
st.info("STEP 1: Mempersiapkan environment dan memuat data...") # Gunakan st.info

# Install Sastrawi & Imbalanced-learn (jika belum ada)
# Di Streamlit Cloud, dependensi biasanya diatur di requirements.txt
# Baris !pip ini mungkin tidak diperlukan jika sudah diatur di requirements.txt
# print("   - Menginstall library Sastrawi dan Imbalanced-learn...")
# !pip install Sastrawi imbalanced-learn -q
# print("     Selesai install.")

# Mengimpor library sudah dilakukan di atas

# Mengabaikan warning tertentu
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- PINDAHKAN DOWNLOAD NLTK KE SINI ---
st.info("   - Mengecek dan men-download resource NLTK 'punkt'...") # Gunakan st.info
try:
    nltk.data.find('tokenizers/punkt') # Cek langsung resource
    st.success("     Resource 'punkt' sudah ada.") # Gunakan st.success
except LookupError:
    st.warning("     Resource 'punkt' tidak ditemukan, men-download...") # Gunakan st.warning
    try:
        nltk.download('punkt', quiet=True)
        st.success("     Selesai download NLTK 'punkt'.") # Gunakan st.success
    except Exception as e:
        st.error(f"     Gagal download NLTK 'punkt': {e}") # Gunakan st.error
# --- AKHIR PEMINDAHAN DOWNLOAD ---

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
# (Fungsi clean_text, normalize_tokens, remove_stopwords, stem_tokens tetap sama)
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text); text = re.sub(r'@\w+', '', text); text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip(); return text

def normalize_tokens(tokens, kamus_normalisasi):
    if not isinstance(tokens, list): return []
    return [kamus_normalisasi.get(word, word) for word in tokens]

def remove_stopwords(tokens, stop_words):
    if not isinstance(tokens, list): return []
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def stem_tokens(tokens, _stemmer):
    if not isinstance(tokens, list): return []
    try:
        stemmed_list = [_stemmer.stem(token) for token in tokens]
        return [word for word in stemmed_list if word]
    except Exception as e:
        return tokens

# ==============================================================================
# "Canggih" Labeling Function (Tetap sama)
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
                 prev_word = words[i-1]
                 booster_score = senti_dict.get(prev_word, 0)
                 if booster_score in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                     if score > 0: score += booster_score
                     elif score < 0: score -= booster_score
            if score > 0:
                pos_score += score
            else:
                neg_score += abs(score)
            is_negated = False
    if pos_score == 0 and neg_score == 0: return 'netral'
    if pos_score > neg_score * 1.5: return 'positif'
    elif neg_score > pos_score * 1.5: return 'negatif'
    else: return 'netral'

# ==============================================================================
# Preprocessing + Stemming + Labeling Function (Cacheable - PERBAIKAN NLTK CHECK)
# ==============================================================================
@st.cache_data(show_spinner=False)
def full_preprocess_and_label_df(_df, text_column, _kamus_normalisasi, _stop_words, _stemmer, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    """Applies all preprocessing steps including stemming and lexicon labeling."""
    st.info("Memulai preprocessing lengkap & pelabelan lexicon...")
    df_processed = _
