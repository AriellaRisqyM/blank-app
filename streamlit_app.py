# ==============================================================================
# Aplikasi Streamlit: Analisis Sentimen (Lexicon + ML)
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

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ! ! ! IMPORT BARU UNTUK WORDCLOUD ! ! !
from wordcloud import WordCloud

# Configure tqdm for Streamlit
tqdm.pandas()

# ==============================================================================
# Cache Resource Loading Functions (Dictionaries, Models)
# ==============================================================================
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
    try: factory=StopWordRemoverFactory(); sw_sastrawi=set(factory.get_stop_words()); st.success(f"Ok: {len(sw_sastrawi)} stopwords Sastrawi.")
    except Exception as e: st.warning(f"Sastrawi gagal: {e}.")
    try: response=requests.get(url_kamus_online); response.raise_for_status(); sw_kamus=set(response.text.splitlines()); st.success(f"Ok: {len(sw_kamus)} stopwords online.")
    except Exception as e: st.warning(f"Stopwords online gagal: {e}.")
    combined = sw_sastrawi.union(sw_kamus); st.success(f"Total stopwords: {len(combined)}."); return combined
@st.cache_resource
def get_stemmer(): st.info("Inisialisasi stemmer..."); factory=StemmerFactory(); stemmer=factory.create_stemmer(); st.success("Stemmer siap."); return stemmer
@st.cache_resource
def load_tsv_dict(url, key_col, val_col, header=None, sep='\t'):
    try:
        df=pd.read_csv(url, sep=sep, header=header, names=[key_col, val_col], on_bad_lines='skip', engine='python'); df[val_col]=df[val_col].astype(str).str.replace(r'[+]', '', regex=True)
        valid_rows=pd.to_numeric(df[val_col], errors='coerce').notna(); df_valid=df[valid_rows]; return dict(zip(df_valid[key_col], df_valid[val_col].astype(int)))
    except Exception as e: st.warning(f"Gagal TSV {url.split('/')[-1]}: {e}"); return {}
@st.cache_resource
def load_manual_dict(url):
    dictionary={};
    try:
        r=requests.get(url); r.raise_for_status(); lines=r.text.splitlines()
        for line in lines:
            line=line.strip();
            if not line or line.startswith(('word ', 'phrase ', 'emo ')): continue
            parts=line.rsplit(' ', 1);
            if len(parts)==2:
                key, val=parts;
                try: score=int(val.strip().replace('+', '')); dictionary[key.strip()]=score
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
    senti_dict.update(load_manual_dict(urls['emoticon'])); senti_dict.update(load_manual_dict(urls['booster']))
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
    text = re.sub(r'[^a-zA-Z\s]', '', text); text = re.sub(r'\s+', ' ', text).strip(); return text

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
# "Canggih" Labeling Function
# ==============================================================================
def label_sentiment_canggih(text, senti_dict, sorted_idioms, negating_words, question_words):
    if not isinstance(text, str) or not text.strip(): return 'netral'
    original_text_lower = text.lower(); pos_score = 0; neg_score = 0
    words_for_q_check = original_text_lower.split()
    if any(q_word in words_for_q_check for q_word in question_words): return 'netral'
    text_after_idioms = original_text_lower; processed_idiom = False
    for idiom, score in sorted_idioms:
        if f" {idiom} " in f" {text_after_idioms} ":
            if score > 0: pos_score += score
            else: neg_score += abs(score)
            text_after_idioms = text_after_idioms.replace(idiom, " "); processed_idiom = True
    words = text_after_idioms.split() if processed_idiom else words_for_q_check
    is_negated = False
    for i, word in enumerate(words):
        if not word: continue
        if word in negating_words: is_negated = True; continue
        score = senti_dict.get(word, 0)
        if score != 0:
            if is_negated: score *= -1
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
# Preprocessing + Labeling Function (Cacheable)
# ==============================================================================
@st.cache_data
def preprocess_and_label_df(_df, text_column, _senti_dict, _sorted_idioms, _negating_words, _question_words):
    """Applies cleaning, case folding, and lexicon labeling."""
    st.info("Memulai preprocessing & pelabelan lexicon...")
    df_processed = _df.copy() # Use _df to avoid conflict with streamlit elements

    # Apply Cleaning & Case Folding
    st.write("1. Cleaning & Case Folding...")
    df_processed['cleaned_text'] = df_processed[text_column].apply(clean_text)
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    if df_processed.empty:
        st.warning("Tidak ada teks valid setelah cleaning.")
        return df_processed # Return empty df
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()

    # Apply "Canggih" Labeling on 'case_folded_text'
    st.write("2. Pelabelan Sentimen (Lexicon Canggih)...")
    # Make sure to pass all required dictionaries
    df_processed['sentiment'] = df_processed['case_folded_text'].progress_apply(
        lambda text: label_sentiment_canggih(text, _senti_dict, _sorted_idioms, _negating_words, _question_words)
    )

    st.success("Preprocessing & Pelabelan Lexicon Selesai.")
    # Return df with key columns first
    cols_to_return = [text_column, 'case_folded_text', 'sentiment'] + [col for col in df_processed.columns if col not in [text_column, 'case_folded_text', 'sentiment', 'cleaned_text']]
    return df_processed[cols_to_return]

# ==============================================================================
# ML Modeling Function (Not Cached for dynamic training)
# ==============================================================================
def train_and_evaluate_models(df_labeled, selected_test_size=0.3, text_col_for_tfidf='case_folded_text', target_col='sentiment'):
    """Performs Train/Test split, TF-IDF, trains NB & SVM, returns results."""
    st.info("Memulai persiapan data dan pelatihan model ML...")

    if df_labeled.empty or target_col not in df_labeled.columns or text_col_for_tfidf not in df_labeled.columns:
        st.error("Dataframe input kosong atau kolom yang dibutuhkan tidak ada.")
        return None

    # Pastikan tidak ada NaN di kolom teks atau target
    df_labeled = df_labeled.dropna(subset=[text_col_for_tfidf, target_col])
    if len(df_labeled[target_col].unique()) < 2:
         st.error(f"Hanya ada {len(df_labeled[target_col].unique())} kelas unik di kolom target '{target_col}'. Minimal 2 kelas dibutuhkan untuk pelatihan.")
         return None
    if len(df_labeled) < 10: # Arbitrary small number
         st.warning(f"Jumlah data ({len(df_labeled)}) sangat sedikit, hasil model mungkin tidak reliable.")


    X = df_labeled[text_col_for_tfidf]
    y = df_labeled[target_col]

    # Split Data (Stratify to handle imbalance)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=selected_test_size, # <-- Menggunakan nilai dari slider
            random_state=42,
            stratify=y
        )
        
        st.write(f"Data dibagi: {len(X_train)} train, {len(X_test)} test (Test Size = {selected_test_size})")
        st.write("Distribusi kelas (Train):", y_train.value_counts())
        st.write("Distribusi kelas (Test):", y_test.value_counts())
        
    except ValueError as e:
         st.error(f"Gagal membagi data (mungkin karena data terlalu sedikit atau hanya 1 kelas di split): {e}")
         return None


    # TF-IDF Vectorization
    st.write("Membuat fitur TF-IDF...")
    # Limit features for performance, can be adjusted
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    st.write(f"Bentuk matriks TF-IDF (Train): {X_train_tfidf.shape}")

    results = {}

    # Naive Bayes
    st.write("Melatih Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.2) # Alpha from notebook
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    nb_report = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)
    nb_cm = confusion_matrix(y_test, y_pred_nb, labels=nb_model.classes_)
    results['naive_bayes'] = {'accuracy': nb_accuracy, 'report': nb_report, 'cm': nb_cm, 'labels': nb_model.classes_}
    st.write(f"Akurasi Naive Bayes: {nb_accuracy*100:.2f}%")

    # SVM
    st.write("Melatih SVM (Linear)...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_report = classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)
    svm_cm = confusion_matrix(y_test, y_pred_svm, labels=svm_model.classes_) # Use model classes for label order
    results['svm'] = {'accuracy': svm_accuracy, 'report': svm_report, 'cm': svm_cm, 'labels': svm_model.classes_}
    st.write(f"Akurasi SVM: {svm_accuracy*100:.2f}%")

    st.success("Pelatihan dan evaluasi model ML selesai.")
    return results


# ==============================================================================
# Single Text Processing Function
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

# ! ! ! FUNGSI BARU UNTUK WORDCLOUD ! ! !
@st.cache_data
def generate_wordcloud(_df, sentiment_label, _stop_words):
    """Membuat dan mengembalikan array gambar Word Cloud."""
    st.info(f"Membuat word cloud untuk sentimen '{sentiment_label}'...")
    text_data = _df[_df['sentiment'] == sentiment_label]['case_folded_text']
    
    if text_data.empty:
        st.write(f"Tidak ada data '{sentiment_label}' untuk Word Cloud.")
        return None
        
    all_text = " ".join(text for text in text_data)
    
    if not all_text.strip():
        st.write(f"Tidak ada teks valid '{sentiment_label}' untuk Word Cloud.")
        return None
    
    try:
        # Tentukan colormap berdasarkan sentimen
        colormap = 'Greens' if sentiment_label == 'positif' else ('Reds' if sentiment_label == 'negatif' else 'Greys')
        
        wordcloud = WordCloud(width=800, height=400, 
                              background_color='white', 
                              stopwords=_stop_words, 
                              colormap=colormap,  # Menggunakan colormap yang dipilih
                              min_font_size=10,
                              prefer_horizontal=0.9).generate(all_text)
        
        return wordcloud.to_array() # Mengembalikan array untuk st.image
    
    except Exception as e:
        st.warning(f"Gagal membuat word cloud untuk '{sentiment_label}': {e}")
        return None


# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="Analisis Sentimen Lexicon+ML")

st.title("📊 Aplikasi Analisis Sentimen (Lexicon + ML)")
st.markdown("""
Aplikasi ini melakukan preprocessing, pelabelan sentimen berbasis **Lexicon** (InSet & SentiStrength),
dan melatih model **Machine Learning** (Naive Bayes & SVM) menggunakan TF-IDF pada data yang diunggah.

Pilih metode input: **Unggah File** (untuk lexicon & ML) atau **Input Teks Langsung** (hanya lexicon).
""")

# --- Load Resources ---
with st.spinner("⏳ Memuat sumber daya (kamus, model)..."):
    # (URLs remain the same)
    url_kamus_norm = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt'
    url_stopwords_online = 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt'
    lexicon_urls = { 'inset_fajri_pos': 'https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv', 'inset_fajri_neg': 'https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv', 'inset_onpilot_pos': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv', 'inset_onpilot_neg': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv', 'senti_json': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt', 'booster': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/boosterwords_id.txt', 'emoticon': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/emoticon_id.txt', 'idiom': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/idioms_id.txt', 'negating': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/negatingword.txt', 'question': 'https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/questionword.txt' }
    
    # Muat semua resource
    kamus_normalisasi = load_kamus_normalisasi(url_kamus_norm)
    stop_words = load_combined_stopwords(url_stopwords_online) # <--- stop_words dimuat di sini
    stemmer = get_stemmer()
    senti_dict, sorted_idioms, negating_words, question_words = load_all_sentiment_dictionaries(lexicon_urls)
    
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: st.info("Mengunduh NLTK Punkt..."); nltk.download('punkt')

st.success("✅ Sumber daya siap.")
st.markdown("---")

# --- Tabs ---
tab1, tab2 = st.tabs(["📤 Unggah File (Lexicon + ML)", "⌨️ Input Teks Langsung (Hanya Lexicon)"])

# --- Tab 1: File Upload ---
with tab1:
    st.header("1. Unggah Data Anda (CSV/Excel)")
    uploaded_file = st.file_uploader("Pilih file", type=['csv', 'xlsx'], label_visibility="collapsed", key="file_uploader") # <-- Kunci ini sekarang unik

    # Initialize session state for results if not present
    if 'labeled_df' not in st.session_state:
        st.session_state['labeled_df'] = None
    if 'ml_results' not in st.session_state:
        st.session_state['ml_results'] = None
    if 'processed_filename' not in st.session_state:
        st.session_state['processed_filename'] = ""


    if uploaded_file is not None:
        # Clear previous results if a new file is uploaded
        if uploaded_file.name != st.session_state.get('processed_filename', ""):
             st.session_state['labeled_df'] = None
             st.session_state['ml_results'] = None
             st.session_state['processed_filename'] = uploaded_file.name


        try:
            if uploaded_file.name.endswith('.csv'): df_input = pd.read_csv(uploaded_file)
            else: df_input = pd.read_excel(uploaded_file, engine='openpyxl')

            st.success(f"File '{uploaded_file.name}' diunggah ({len(df_input)} baris).")
            st.dataframe(df_input.head(), hide_index=True)

            st.header("2. Pilih Kolom Teks")
            available_columns = [""] + df_input.columns.tolist()
            text_column = st.selectbox("Pilih kolom:", options=available_columns, index=0, key="selectbox_column")

            if text_column:
                st.info(f"Kolom teks: **{text_column}**")

                # --- Preprocessing & Lexicon Labeling Button ---
                st.header("3. Proses Preprocessing & Labeling Lexicon")
                if st.button("🔬 Proses Teks & Label Lexicon", key="button_process_lexicon"):
                    st.session_state['labeled_df'] = None # Clear previous labeling result
                    st.session_state['ml_results'] = None # Clear previous ML result

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
                            with st.spinner("⏳ Memproses & melabel (lexicon)..."):
                                df_labeled_lexicon = preprocess_and_label_df(
                                    df_valid, text_column,
                                    senti_dict, sorted_idioms, negating_words, question_words
                                )
                            if df_labeled_lexicon.empty: st.warning("Tidak ada hasil labeling.")
                            else:
                                st.session_state['labeled_df'] = df_labeled_lexicon # Store result

                # --- Display Lexicon Labeling Results ---
                if st.session_state['labeled_df'] is not None:
                    st.success("🎉 Preprocessing & Labeling Lexicon Selesai!")
                    st.subheader("Hasil Labeling Lexicon (Contoh):")
                    st.dataframe(st.session_state['labeled_df'][[text_column, 'case_folded_text', 'sentiment']].head(20), hide_index=True)
                    st.subheader("Distribusi Sentimen (Lexicon):")
                    sentiment_counts_lex = st.session_state['labeled_df']['sentiment'].value_counts()
                    st.bar_chart(sentiment_counts_lex)
                    st.write(sentiment_counts_lex)

                    # ! ! ! BAGIAN BARU UNTUK WORDCLOUD ! ! !
                    st.subheader("Word Clouds Sentimen (Lexicon):")
                    st.write("Word cloud dibuat dari kolom 'case_folded_text' berdasarkan label lexicon.")
                    
                    col_wc1, col_wc2 = st.columns(2)
                    
                    with col_wc1:
                        st.markdown("<h5>Positif 😊</h5>", unsafe_allow_html=True)
                        with st.spinner("Membuat Word Cloud Positif..."):
                            # Memanggil fungsi generate_wordcloud
                            wc_pos_array = generate_wordcloud(
                                st.session_state['labeled_df'], 
                                'positif', 
                                stop_words # <--- Menggunakan stop_words yang sudah dimuat
                            )
                        if wc_pos_array is not None:
                            st.image(wc_pos_array, use_column_width=True)
                        else:
                            st.write("Tidak ada data positif untuk Word Cloud.")

                    with col_wc2:
                        st.markdown("<h5>Negatif 😠</h5>", unsafe_allow_html=True)
                        with st.spinner("Membuat Word Cloud Negatif..."):
                             # Memanggil fungsi generate_wordcloud
                            wc_neg_array = generate_wordcloud(
                                st.session_state['labeled_df'], 
                                'negatif', 
                                stop_words # <--- Menggunakan stop_words yang sudah dimuat
                            )
                        if wc_neg_array is not None:
                            st.image(wc_neg_array, use_column_width=True)
                        else:
                            st.write("Tidak ada data negatif untuk Word Cloud.")
                    
                    st.markdown("---") # Pemisah sebelum ML
                    # ! ! ! AKHIR BAGIAN BARU WORDCLOUD ! ! !


                    # --- ML Modeling Button ---
                    st.header("4. Latih & Evaluasi Model ML (Naive Bayes & SVM)")
                    st.warning("Model ML akan dilatih menggunakan kolom 'sentiment' hasil lexicon sebagai target.")
                    
                    # Slider Test Size (dari permintaan Anda sebelumnya)
                    selected_test_size = st.slider(
                        label="Pilih Test Size (Proporsi Data Uji):",
                        min_value=0.1,  # 10%
                        max_value=0.9,  # 90%
                        value=0.3,      # Default 30%
                        step=0.1,       # Kelipatan 10%
                        key="slider_test_size"
                    )
                    st.info(f"Data uji akan menggunakan {int(selected_test_size*100)}% dari total data. (Data Latih: {int((1-selected_test_size)*100)}%)")
                    
                    
                    if st.button("🤖 Latih Model ML", key="button_train_ml"):
                         st.session_state['ml_results'] = None # Clear previous ML results
                         with st.spinner("⏳ Melatih & mengevaluasi model ML..."):
                            
                             ml_results_dict = train_and_evaluate_models(
                                 st.session_state['labeled_df'],
                                 selected_test_size=selected_test_size # <-- Menggunakan nilai dari slider
                             )
                             
                             if ml_results_dict:
                                 st.session_state['ml_results'] = ml_results_dict # Store results


                    # --- Display ML Results ---
                    if st.session_state['ml_results'] is not None:
                        st.success("🎉 Pelatihan & Evaluasi Model ML Selesai!")
                        results_ml = st.session_state['ml_results']

                        st.subheader("Hasil Evaluasi Model:")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Akurasi Naive Bayes", f"{results_ml['naive_bayes']['accuracy']:.2%}")
                            st.text("Laporan Klasifikasi NB:")
                            st.dataframe(pd.DataFrame(results_ml['naive_bayes']['report']).transpose())
                        with col2:
                            st.metric("Akurasi SVM (Linear)", f"{results_ml['svm']['accuracy']:.2%}")
                            st.text("Laporan Klasifikasi SVM:")
                            st.dataframe(pd.DataFrame(results_ml['svm']['report']).transpose())

                        st.subheader("Confusion Matrix:")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        sns.heatmap(results_ml['naive_bayes']['cm'], annot=True, fmt='d', cmap='Blues', ax=ax1,
                                    xticklabels=results_ml['naive_bayes']['labels'], yticklabels=results_ml['naive_bayes']['labels'])
                        ax1.set_title('Naive Bayes')
                        ax1.set_xlabel('Predicted')
                        ax1.set_ylabel('Actual')

                        sns.heatmap(results_ml['svm']['cm'], annot=True, fmt='d', cmap='Oranges', ax=ax2,
                                    xticklabels=results_ml['svm']['labels'], yticklabels=results_ml['svm']['labels'])
                        ax2.set_title('SVM (Linear)')
                        ax2.set_xlabel('Predicted')
                        ax2.set_ylabel('Actual')

                        st.pyplot(fig)

                        # --- Download Button for Labeled Data ---
                        st.header("5. Unduh Hasil (Termasuk Label Lexicon)")
                        csv_data = convert_df_to_csv(st.session_state['labeled_df'])
                        output_filename = f"hasil_sentimen_lexicon_{uploaded_file.name.split('.')[0]}.csv"
                        st.download_button(
                            label="📥 Unduh Data + Label Lexicon (.csv)", data=csv_data,
                            file_name=output_filename, mime='text/csv', key="download_button"
                        )

                elif not text_column:
                     st.warning("☝️ Pilih kolom teks untuk memulai proses.")


        except Exception as e:
            st.error(f"Error membaca atau memproses file: {e}")

    else:
        st.info("Unggah file CSV atau Excel di atas untuk memulai.")


# --- Tab 2: Text Input ---
with tab2:
    st.header("Analisis Teks Tunggal (Hanya Lexicon)")
    st.info("Fitur ini hanya menggunakan metode berbasis kamus (lexicon), tidak melatih model ML.")
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
                 st.session_state['processed_text_result'] = {'text': user_text, 'processed': teks_diproses, 'sentiment': hasil_sentimen}
            else:
                 st.error(f"Gagal: {teks_diproses}")
        else:
            st.warning("☝️ Masukkan teks terlebih dahulu.")

# --- Footer ---
st.markdown("---")
st.markdown("Dibuat dengan Streamlit")
