# ==============================================================================
# Aplikasi Streamlit untuk Analisis Sentimen Berbasis Leksikon (File & Teks)
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
import io # Required for download button

# Configure tqdm for Streamlit
tqdm.pandas()

# ==============================================================================
# Cache Resource Loading Functions (Dictionaries, Models) - SAME AS BEFORE
# ==============================================================================

@st.cache_resource
def load_kamus_normalisasi(url):
    st.info(f"Mengunduh kamus normalisasi dari {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        kamus = response.json()
        st.success(f"Berhasil memuat {len(kamus)} kata normalisasi.")
        return kamus
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal memuat kamus normalisasi: {e}")
    except json.JSONDecodeError:
        st.error(f"Gagal memparsing JSON kamus normalisasi.")
    st.warning("Menggunakan kamus normalisasi bawaan (darurat)...")
    return {
        'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'tdk': 'tidak', 'gaada': 'tidak ada',
        'bgt': 'banget', 'jgn': 'jangan', 'krn': 'karena', 'utk': 'untuk',
        'lg': 'lagi', 'sm': 'sama', 'kalo': 'kalau', 'udah': 'sudah'
    }

@st.cache_resource
def load_combined_stopwords(url_kamus_online):
    stop_words_sastrawi = set()
    stop_words_kamus = set()
    st.info("Memuat stopwords Sastrawi...")
    try:
        factory = StopWordRemoverFactory()
        stop_words_sastrawi = set(factory.get_stop_words())
        st.success(f"Berhasil memuat {len(stop_words_sastrawi)} stopwords Sastrawi.")
    except Exception as e:
        st.warning(f"Gagal memuat Sastrawi: {e}.")

    st.info(f"Mengunduh stopwords online dari {url_kamus_online}...")
    try:
        response = requests.get(url_kamus_online)
        response.raise_for_status()
        stop_words_kamus = set(response.text.splitlines())
        st.success(f"Berhasil memuat {len(stop_words_kamus)} stopwords online.")
    except requests.exceptions.RequestException as e:
        st.warning(f"Gagal memuat stopwords online: {e}.")

    combined_stopwords = stop_words_sastrawi.union(stop_words_kamus)
    st.success(f"Total stopwords gabungan: {len(combined_stopwords)} kata unik.")
    return combined_stopwords

@st.cache_resource
def get_stemmer():
    st.info("Inisialisasi stemmer Sastrawi...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    st.success("Stemmer siap.")
    return stemmer

@st.cache_resource
def load_tsv_dict(url, key_col, val_col, header=None, sep='\t'):
    try:
        df = pd.read_csv(url, sep=sep, header=header, names=[key_col, val_col], on_bad_lines='skip', engine='python')
        df[val_col] = df[val_col].astype(str).str.replace(r'[+]', '', regex=True)
        valid_rows = pd.to_numeric(df[val_col], errors='coerce').notna()
        df_valid = df[valid_rows]
        return dict(zip(df_valid[key_col], df_valid[val_col].astype(int)))
    except Exception as e:
        st.warning(f"Gagal memuat TSV {url}: {e}")
        return {}

@st.cache_resource
def load_manual_dict(url):
    dictionary = {}
    try:
        r = requests.get(url)
        r.raise_for_status()
        lines = r.text.splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('word ', 'phrase ', 'emo ')): continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                key, val = parts
                try:
                    score = int(val.strip().replace('+', ''))
                    dictionary[key.strip()] = score
                except ValueError: pass
        return dictionary
    except Exception as e:
        st.warning(f"Gagal memuat manual {url}: {e}")
        return {}

@st.cache_resource
def load_set(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return set(line.strip() for line in r.text.splitlines() if line.strip())
    except Exception as e:
        st.warning(f"Gagal memuat set {url}: {e}")
        return set()

@st.cache_resource
def load_all_sentiment_dictionaries(urls):
    st.info("Memuat semua kamus sentimen...")
    senti_dict = {}
    senti_dict.update(load_tsv_dict(urls['inset_fajri_pos'], 'word', 'score', header=None, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_fajri_neg'], 'word', 'score', header=None, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_onpilot_pos'], 'word', 'weight', header=0, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_onpilot_neg'], 'word', 'weight', header=0, sep='\t'))
    try:
        r = requests.get(urls['senti_json'])
        r.raise_for_status()
        senti_dict.update(r.json())
        st.info("Berhasil memuat SentiStrength JSON.")
    except Exception as e:
        st.warning(f"Gagal memuat Senti JSON: {e}")
    senti_dict.update(load_manual_dict(urls['emoticon']))
    senti_dict.update(load_manual_dict(urls['booster']))
    idiom_dict = load_manual_dict(urls['idiom'])
    sorted_idioms = sorted(idiom_dict.items(), key=lambda x: len(x[0]), reverse=True)
    negating_words = load_set(urls['negating'])
    question_words = load_set(urls['question'])

    st.success(f"Total item kamus sentimen utama: {len(senti_dict)}")
    st.success(f"Total idiom: {len(idiom_dict)}")
    st.success(f"Total kata negasi: {len(negating_words)}")
    st.success(f"Total kata tanya: {len(question_words)}")
    return senti_dict, sorted_idioms, negating_words, question_words

# ==============================================================================
# Preprocessing Functions - SAME AS BEFORE
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ... (normalize_tokens, remove_stopwords, stem_tokens - SAME AS BEFORE) ...
def normalize_tokens(tokens, kamus_normalisasi):
    if not isinstance(tokens, list): return []
    return [kamus_normalisasi.get(word, word) for word in tokens]

def remove_stopwords(tokens, stop_words):
    if not isinstance(tokens, list): return []
    return [word for word in tokens if word not in stop_words and len(word) > 1]

def stem_tokens(tokens, _stemmer):
    if not isinstance(tokens, list): return []
    return [stemmed for stemmed in [_stemmer.stem(word) for word in tokens] if stemmed]


# ==============================================================================
# "Canggih" Labeling Function - SAME AS BEFORE
# ==============================================================================
def label_sentiment_canggih(text, senti_dict, sorted_idioms, negating_words, question_words):
    if not isinstance(text, str) or not text.strip():
        return 'netral'
    original_text_lower = text.lower()
    pos_score = 0
    neg_score = 0
    words_for_q_check = original_text_lower.split()
    if any(q_word in words_for_q_check for q_word in question_words):
         return 'netral'
    text_after_idioms = original_text_lower
    processed_idiom = False
    for idiom, score in sorted_idioms:
        if f" {idiom} " in f" {text_after_idioms} ":
            if score > 0: pos_score += score
            else: neg_score += abs(score)
            text_after_idioms = text_after_idioms.replace(idiom, " ")
            processed_idiom = True
    words = text_after_idioms.split() if processed_idiom else words_for_q_check
    is_negated = False
    for i, word in enumerate(words):
        if not word: continue
        if word in negating_words:
            is_negated = True
            continue
        score = senti_dict.get(word, 0)
        if score != 0:
            if is_negated:
                score *= -1
            if i > 0 and words[i-1] in senti_dict:
                 booster_score = senti_dict.get(words[i-1], 0)
                 if booster_score in [-2, -1, 1, 2]:
                     if score > 0: score += booster_score
                     elif score < 0: score -= booster_score
            if score > 0: pos_score += score
            else: neg_score += abs(score)
            is_negated = False
    if pos_score == 0 and neg_score == 0: return 'netral'
    if pos_score > neg_score * 1.5: return 'positif'
    elif neg_score > pos_score * 1.5: return 'negatif'
    else: return 'netral'

# ==============================================================================
# Main Processing Function - MODIFIED FOR SINGLE TEXT
# ==============================================================================

# Keep the original for files
@st.cache_data
def process_dataframe(df, text_column, _kamus_normalisasi, _stop_words, _stemmer, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    st.info("Memulai preprocessing & pelabelan file...")
    df_processed = df.copy()
    st.write("1. Cleaning & Case Folding...")
    df_processed['cleaned_text'] = df_processed[text_column].apply(clean_text)
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    if df_processed.empty: return df_processed
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()
    st.write("2. Pelabelan Sentimen (Canggih)...")
    df_processed['sentiment'] = df_processed['case_folded_text'].progress_apply(
        lambda text: label_sentiment_canggih(text, _senti_dict, _sorted_idioms, _negating_words, _question_words)
    )
    st.success("Preprocessing & Pelabelan File Selesai.")
    # Return df with original columns + new ones, reordering slightly
    return df_processed[[text_column,'case_folded_text', 'sentiment']].join(df_processed.drop([text_column,'case_folded_text', 'sentiment', 'cleaned_text'], axis=1, errors='ignore'))


# NEW function for single text input
def process_single_text(input_text, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    """Cleans, case folds, and labels a single string input."""
    if not input_text or not isinstance(input_text, str) or not input_text.strip():
        return None, "Input teks kosong atau tidak valid."

    cleaned = clean_text(input_text)
    if not cleaned:
        return None, "Teks menjadi kosong setelah cleaning."

    case_folded = cleaned.lower()
    sentiment = label_sentiment_canggih(case_folded, _senti_dict, _sorted_idioms, _negating_words, _question_words)
    return sentiment, case_folded # Return sentiment and the text used for labeling

# ==============================================================================
# Helper Function for Download - SAME AS BEFORE
# ==============================================================================
@st.cache_data
def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# Streamlit UI - MODIFIED WITH TABS
# ==============================================================================

st.set_page_config(layout="wide", page_title="Analisis Sentimen Lexicon")

st.title("📊 Aplikasi Analisis Sentimen Berbasis Leksikon")
st.markdown("""
Aplikasi ini melakukan preprocessing teks dan **pelabelan sentimen** (Positif, Negatif, Netral)
menggunakan kamus leksikon **InSet & SentiStrength** yang dimuat otomatis.

Pilih metode input: **Unggah File** atau **Input Teks Langsung**.
""")

# --- Load Resources (runs only once thanks to cache) ---
with st.spinner("⏳ Memuat sumber daya (kamus, model)... Ini mungkin perlu beberapa saat."):
    url_kamus_norm = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt'
    url_stopwords_online = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt'
    lexicon_urls = {
        'inset_fajri_pos': 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv',
        'inset_fajri_neg': 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv',
        'inset_onpilot_pos': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv',
        'inset_onpilot_neg': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv',
        'senti_json': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt',
        'booster': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/boosterwords_id.txt',
        'emoticon': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/emoticon_id.txt',
        'idiom': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/idioms_id.txt',
        'negating': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/negatingword.txt',
        'question': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/questionword.txt'
    }
    kamus_normalisasi = load_kamus_normalisasi(url_kamus_norm)
    stop_words = load_combined_stopwords(url_stopwords_online)
    stemmer = get_stemmer()
    senti_dict, sorted_idioms, negating_words, question_words = load_all_sentiment_dictionaries(lexicon_urls)
    try: nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Mengunduh NLTK Punkt tokenizer (pertama kali)...")
        nltk.download('punkt')
st.success("✅ Sumber daya siap.")
st.markdown("---")


# --- Tabs for Input Method ---
tab1, tab2 = st.tabs(["📤 Unggah File", "⌨️ Input Teks Langsung"])

# --- Tab 1: File Upload ---
with tab1:
    st.header("1. Unggah Data Anda (CSV/Excel)")
    uploaded_file = st.file_uploader("Pilih file", type=['csv', 'xlsx'], label_visibility="collapsed", key="file_uploader")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else: # .xlsx
                df_input = pd.read_excel(uploaded_file, engine='openpyxl')

            st.success(f"File '{uploaded_file.name}' diunggah ({len(df_input)} baris).")
            st.dataframe(df_input.head(), hide_index=True)

            st.header("2. Pilih Kolom Teks")
            available_columns = [""] + df_input.columns.tolist()
            text_column = st.selectbox(
                "Pilih kolom berisi teks:",
                options=available_columns, index=0, key="selectbox_column"
                )

            if text_column:
                st.info(f"Kolom teks: **{text_column}**")
                st.header("3. Proses Analisis Sentimen File")
                if st.button("🚀 Proses File!", key="button_process_file"):
                    if text_column not in df_input.columns: st.error("Kolom tidak valid.")
                    elif df_input[text_column].isnull().all(): st.error(f"Kolom '{text_column}' kosong.")
                    else:
                        if not pd.api.types.is_string_dtype(df_input[text_column]):
                             try:
                                 df_input[text_column] = df_input[text_column].astype(str)
                                 st.warning(f"Kolom '{text_column}' dikonversi ke teks.")
                             except Exception as e:
                                 st.error(f"Gagal konversi kolom: {e}"); st.stop()

                        df_valid = df_input.dropna(subset=[text_column])
                        df_valid = df_valid[df_valid[text_column].astype(str).str.strip().astype(bool)]

                        if df_valid.empty: st.warning("Tidak ada teks valid.")
                        else:
                            with st.spinner("⏳ Memproses file..."):
                                df_output = process_dataframe(
                                    df_valid, text_column,
                                    kamus_normalisasi, stop_words, stemmer,
                                    senti_dict, sorted_idioms, negating_words, question_words
                                )
                            if df_output.empty: st.warning("Tidak ada hasil.")
                            else:
                                st.success("🎉 Proses file selesai!")
                                st.header("4. Hasil Analisis File")
                                st.dataframe(df_output.head(50), hide_index=True)
                                st.subheader("Distribusi Sentimen:")
                                sentiment_counts = df_output['sentiment'].value_counts()
                                st.bar_chart(sentiment_counts)
                                st.write(sentiment_counts)
                                st.header("5. Unduh Hasil File")
                                csv_data = convert_df_to_csv(df_output)
                                output_filename = f"hasil_sentimen_{uploaded_file.name.split('.')[0]}.csv"
                                st.download_button(
                                    label="📥 Unduh Hasil (.csv)", data=csv_data,
                                    file_name=output_filename, mime='text/csv', key="download_button"
                                )
                                st.session_state['processed_df_file'] = df_output # Store if needed
            else:
                st.warning("☝️ Pilih kolom teks untuk melanjutkan.")
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    else:
        st.info("Unggah file CSV atau Excel di atas.")


# --- Tab 2: Text Input ---
with tab2:
    st.header("1. Masukkan Teks")
    user_text = st.text_area("Ketik atau paste teks Anda di sini:", height=150, key="text_area_input")

    st.header("2. Proses Analisis Sentimen Teks")
    if st.button("🚀 Analisis Teks Ini!", key="button_process_text"):
        if user_text and user_text.strip():
            with st.spinner("⏳ Menganalisis teks..."):
                # Call the single text processing function
                hasil_sentimen, teks_diproses = process_single_text(
                    user_text, senti_dict, sorted_idioms, negating_words, question_words
                    )

            st.header("3. Hasil Analisis Teks")
            if hasil_sentimen:
                 st.write("Teks setelah Cleaning & Case Folding:")
                 st.info(f"`{teks_diproses}`") # Show the text used for labeling
                 st.write("Hasil Sentimen:")
                 if hasil_sentimen == 'positif':
                     st.success(f"**{hasil_sentimen.upper()}** 😊")
                 elif hasil_sentimen == 'negatif':
                     st.error(f"**{hasil_sentimen.upper()}** 😠")
                 else:
                     st.warning(f"**{hasil_sentimen.upper()}** 😐")
                 # Store result if needed
                 st.session_state['processed_text_result'] = {'text': user_text, 'processed': teks_diproses, 'sentiment': hasil_sentimen}
            else:
                 st.error(f"Gagal memproses teks: {teks_diproses}") # teks_diproses contains error msg here
        else:
            st.warning("☝️ Harap masukkan teks terlebih dahulu.")

# --- Footer ---
st.markdown("---")
st.markdown("Dibuat dengan Streamlit | Analisis Sentimen Lexicon Otomatis")
