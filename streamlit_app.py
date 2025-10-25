# ==============================================================================
# Aplikasi Streamlit untuk Analisis Sentimen Berbasis Leksikon
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
# Cache Resource Loading Functions (Dictionaries, Models)
# ==============================================================================

@st.cache_resource
def load_kamus_normalisasi(url):
    """Downloads and loads the normalization dictionary from a URL."""
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
    """Loads stopwords from Sastrawi and an online dictionary, then combines them."""
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
    """Initializes and returns the Sastrawi stemmer."""
    st.info("Inisialisasi stemmer Sastrawi...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    st.success("Stemmer siap.")
    return stemmer

@st.cache_resource
def load_tsv_dict(url, key_col, val_col, header=None, sep='\t'): # Default header=None for most InSet
    """Loads a dictionary from a TSV/CSV URL using Pandas."""
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
    """Loads SentiStrength-style dictionaries manually, handling phrases and dirtiness."""
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
                except ValueError: pass # Skip non-integer scores like 'alay:-4'
        return dictionary
    except Exception as e:
        st.warning(f"Gagal memuat manual {url}: {e}")
        return {}

@st.cache_resource
def load_set(url):
    """Loads a set of words (one per line) from a URL."""
    try:
        r = requests.get(url)
        r.raise_for_status()
        return set(line.strip() for line in r.text.splitlines() if line.strip()) # Ensure no empty strings
    except Exception as e:
        st.warning(f"Gagal memuat set {url}: {e}")
        return set()

@st.cache_resource
def load_all_sentiment_dictionaries(urls):
    """Loads all dictionaries needed for the 'Canggih' labeling."""
    st.info("Memuat semua kamus sentimen...")
    senti_dict = {}
    # Load InSet fajri91 (No header in original file)
    senti_dict.update(load_tsv_dict(urls['inset_fajri_pos'], 'word', 'score', header=None, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_fajri_neg'], 'word', 'score', header=None, sep='\t'))
    # Load InSet onpilot (Has header: word<tab>weight)
    senti_dict.update(load_tsv_dict(urls['inset_onpilot_pos'], 'word', 'weight', header=0, sep='\t'))
    senti_dict.update(load_tsv_dict(urls['inset_onpilot_neg'], 'word', 'weight', header=0, sep='\t'))
    # Load SentiStrength JSON (Cleanest source)
    try:
        r = requests.get(urls['senti_json'])
        r.raise_for_status()
        senti_dict.update(r.json())
        st.info("Berhasil memuat SentiStrength JSON.")
    except Exception as e:
        st.warning(f"Gagal memuat Senti JSON: {e}")
    # Load Manual Dictionaries (Emoticon, Booster)
    senti_dict.update(load_manual_dict(urls['emoticon']))
    senti_dict.update(load_manual_dict(urls['booster']))
    # Load Idioms
    idiom_dict = load_manual_dict(urls['idiom'])
    sorted_idioms = sorted(idiom_dict.items(), key=lambda x: len(x[0]), reverse=True)
    # Load Modifiers
    negating_words = load_set(urls['negating'])
    question_words = load_set(urls['question'])

    st.success(f"Total item kamus sentimen utama: {len(senti_dict)}")
    st.success(f"Total idiom: {len(idiom_dict)}")
    st.success(f"Total kata negasi: {len(negating_words)}")
    st.success(f"Total kata tanya: {len(question_words)}")

    return senti_dict, sorted_idioms, negating_words, question_words

# ==============================================================================
# Preprocessing Functions
# ==============================================================================

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_tokens(tokens, kamus_normalisasi):
    if not isinstance(tokens, list): return []
    return [kamus_normalisasi.get(word, word) for word in tokens]

def remove_stopwords(tokens, stop_words):
    if not isinstance(tokens, list): return []
    return [word for word in tokens if word not in stop_words and len(word) > 1]

def stem_tokens(tokens, _stemmer): # Changed stemmer arg name to avoid conflict
    if not isinstance(tokens, list): return []
    return [stemmed for stemmed in [_stemmer.stem(word) for word in tokens] if stemmed]

# ==============================================================================
# "Canggih" Labeling Function
# ==============================================================================

def label_sentiment_canggih(text, senti_dict, sorted_idioms, negating_words, question_words):
    if not isinstance(text, str) or not text.strip():
        return 'netral'

    original_text_lower = text.lower()

    pos_score = 0
    neg_score = 0

    # Cek 1: Kalimat tanya?
    words_for_q_check = original_text_lower.split()
    if any(q_word in words_for_q_check for q_word in question_words):
         return 'netral'

    # Cek 2: Idiom
    text_after_idioms = original_text_lower
    processed_idiom = False
    for idiom, score in sorted_idioms:
        # More robust idiom check needed? This might be slow on large texts.
        # Check if idiom exists as a whole word/phrase
        # Using regex might be better for exact word boundaries: r'\b' + re.escape(idiom) + r'\b'
        # Simplified check for now:
        if f" {idiom} " in f" {text_after_idioms} ":
            if score > 0: pos_score += score
            else: neg_score += abs(score)
            text_after_idioms = text_after_idioms.replace(idiom, " ") # Replace idiom with space
            processed_idiom = True

    # Re-split if idioms were processed, otherwise use original split
    words = text_after_idioms.split() if processed_idiom else words_for_q_check


    # Cek 3: Iterasi kata (skor, negasi, booster)
    is_negated = False
    for i, word in enumerate(words):
        if not word: continue # Skip empty strings resulting from replace

        if word in negating_words:
            is_negated = True
            continue

        score = senti_dict.get(word, 0)

        if score != 0:
            if is_negated:
                score *= -1

            # Simple booster check (previous word)
            if i > 0 and words[i-1] in senti_dict:
                 booster_score = senti_dict.get(words[i-1], 0)
                 # Check if the previous word score indicates it's a booster (e.g., +/- 1 or 2)
                 if booster_score in [-2, -1, 1, 2]:
                     if score > 0: score += booster_score
                     elif score < 0: score -= booster_score # Maintain polarity

            if score > 0: pos_score += score
            else: neg_score += abs(score)

            is_negated = False # Reset negation after finding a scored word
        # else: # If word is not scored, negation might carry over
             # is_negated = False # Simple reset: Negation only affects the immediately following scored word

    # Cek 4: Klasifikasi
    if pos_score == 0 and neg_score == 0:
        return 'netral'
    if pos_score > neg_score * 1.5:
        return 'positif'
    elif neg_score > pos_score * 1.5:
        return 'negatif'
    else:
        return 'netral'

# ==============================================================================
# Main Processing Function (Cache this)
# ==============================================================================

@st.cache_data # Cache results based on input data
def process_data(df, text_column, _kamus_normalisasi, _stop_words, _stemmer, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    """Applies the full preprocessing and labeling pipeline."""
    st.info("Memulai preprocessing & pelabelan...")
    df_processed = df.copy()

    # Apply Cleaning & Case Folding
    st.write("1. Cleaning & Case Folding...")
    df_processed['cleaned_text'] = df_processed[text_column].apply(clean_text)
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    if df_processed.empty:
        st.warning("Tidak ada teks valid setelah cleaning.")
        return df_processed
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()

    # Apply "Canggih" Labeling on 'case_folded_text'
    st.write("2. Pelabelan Sentimen (Canggih)...")
    df_processed['sentiment'] = df_processed['case_folded_text'].progress_apply(
        lambda text: label_sentiment_canggih(text, _senti_dict, _sorted_idioms, _negating_words, _question_words)
    )

    # Note: Further preprocessing steps (tokenizing, normalizing, stopword, stemming)
    # are defined but not applied here as the labeling uses 'case_folded_text'.
    # If you need the fully stemmed text for other purposes, uncomment these lines:
    # st.write("3. Tokenizing...")
    # df_processed['tokenized'] = df_processed['case_folded_text'].apply(nltk.word_tokenize)
    # st.write("4. Normalization...")
    # df_processed['normalized'] = df_processed['tokenized'].apply(lambda t: normalize_tokens(t, _kamus_normalisasi))
    # st.write("5. Stopword Removal...")
    # df_processed['stopwords_removed'] = df_processed['normalized'].apply(lambda t: remove_stopwords(t, _stop_words))
    # st.write("6. Stemming...")
    # df_processed['stemmed_tokens'] = df_processed['stopwords_removed'].progress_apply(lambda t: stem_tokens(t, _stemmer))
    # df_processed['final_processed_text'] = df_processed['stemmed_tokens'].apply(' '.join)

    st.success("Preprocessing & Pelabelan Selesai.")
    return df_processed[['case_folded_text', 'sentiment']].join(df_processed.drop(['case_folded_text', 'sentiment', 'cleaned_text'], axis=1, errors='ignore'))


# ==============================================================================
# Helper Function for Download
# ==============================================================================

@st.cache_data
def convert_df_to_csv(df):
   """Converts DataFrame to CSV bytes for download."""
   return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="Analisis Sentimen Lexicon")

st.title("📊 Aplikasi Analisis Sentimen Berbasis Leksikon")
st.markdown("""
Aplikasi ini melakukan preprocessing teks (cleaning, case folding) dan **pelabelan sentimen** (Positif, Negatif, Netral)
menggunakan kamus leksikon **InSet & SentiStrength** yang dimuat otomatis dari sumber online.

**Unggah file CSV atau Excel Anda** yang berisi kolom teks untuk dianalisis.
""")

# --- Load Resources ---
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
    # These functions are cached
    kamus_normalisasi = load_kamus_normalisasi(url_kamus_norm)
    stop_words = load_combined_stopwords(url_stopwords_online)
    stemmer = get_stemmer() # Although not used in final labeling, keep it loaded if needed later
    senti_dict, sorted_idioms, negating_words, question_words = load_all_sentiment_dictionaries(lexicon_urls)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Mengunduh NLTK Punkt tokenizer (pertama kali)...")
        nltk.download('punkt')
st.success("✅ Sumber daya siap.")

# --- File Upload ---
st.header("1. Unggah Data Anda")
uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=['csv', 'xlsx'], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else: # .xlsx
            df_input = pd.read_excel(uploaded_file, engine='openpyxl')

        st.success(f"File '{uploaded_file.name}' berhasil diunggah ({len(df_input)} baris).")
        st.dataframe(df_input.head(), hide_index=True)

        # --- Column Selection ---
        st.header("2. Pilih Kolom Teks")
        available_columns = [""] + df_input.columns.tolist() # Add empty option for placeholder
        text_column = st.selectbox(
            "Pilih kolom berisi teks:",
            options=available_columns,
            index=0, # Default to empty
            #placeholder="Pilih kolom..." # Older Streamlit versions might not support placeholder
            )

        if text_column:
            st.info(f"Kolom teks yang dipilih: **{text_column}**")

            # --- Processing ---
            st.header("3. Proses Analisis Sentimen")
            if st.button("🚀 Mulai Proses!"):
                # Basic validation
                if text_column not in df_input.columns:
                     st.error("Kolom yang dipilih tidak valid.")
                elif df_input[text_column].isnull().all():
                     st.error(f"Kolom '{text_column}' kosong atau hanya berisi nilai null.")
                else:
                    # Attempt type conversion if needed
                    if not pd.api.types.is_string_dtype(df_input[text_column]):
                         try:
                             df_input[text_column] = df_input[text_column].astype(str)
                             st.warning(f"Kolom '{text_column}' dikonversi menjadi teks.")
                         except Exception as e:
                             st.error(f"Gagal mengonversi kolom '{text_column}' ke teks: {e}")
                             st.stop()

                    # Drop rows with NaN/empty strings in the selected text column
                    df_valid = df_input.dropna(subset=[text_column])
                    df_valid = df_valid[df_valid[text_column].astype(str).str.strip().astype(bool)]

                    if df_valid.empty:
                        st.warning("Tidak ada data teks valid setelah membersihkan baris kosong/null.")
                    else:
                        with st.spinner("⏳ Sedang memproses data... Mohon tunggu."):
                            # Pass loaded resources to the processing function
                            df_output = process_data(
                                df_valid, text_column,
                                kamus_normalisasi, stop_words, stemmer, # Pass loaded resources
                                senti_dict, sorted_idioms, negating_words, question_words
                            )

                        if df_output.empty:
                             st.warning("Tidak ada hasil setelah pemrosesan.")
                        else:
                            st.success("🎉 Proses selesai!")

                            # --- Display Results ---
                            st.header("4. Hasil Analisis")
                            st.dataframe(df_output.head(50), hide_index=True) # Display relevant columns

                            st.subheader("Distribusi Sentimen:")
                            sentiment_counts = df_output['sentiment'].value_counts()
                            try:
                                # Use columns argument for category ordering if needed
                                st.bar_chart(sentiment_counts)
                            except: # Fallback for potential errors with specific data shapes
                                st.write(sentiment_counts)
                            st.write(sentiment_counts)

                            # --- Download Button ---
                            st.header("5. Unduh Hasil")
                            csv_data = convert_df_to_csv(df_output)
                            output_filename = f"hasil_sentimen_{uploaded_file.name.split('.')[0]}.csv"
                            st.download_button(
                                label="📥 Unduh Hasil (.csv)",
                                data=csv_data,
                                file_name=output_filename,
                                mime='text/csv',
                            )
                            # Store result in session state (optional)
                            st.session_state['processed_df'] = df_output
        else:
            st.warning("☝️ Silakan pilih kolom teks untuk melanjutkan.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.error("Pastikan format file benar dan coba lagi.")

else:
    st.info("Silakan unggah file CSV atau Excel Anda untuk memulai.")

# --- Footer ---
st.markdown("---")
st.markdown("Dibuat dengan Streamlit | Analisis Sentimen Lexicon Otomatis")
