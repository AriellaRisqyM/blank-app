# =====================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML) — FINAL 2 KELAS
# =====================================================================
import streamlit as st
import pandas as pd
import requests
import re
import json
import io # Diperlukan untuk membaca data dari requests
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Inisialisasi tqdm untuk pandas (meskipun di Streamlit mungkin tidak tampil di progress_apply)
tqdm.pandas()

# Konfigurasi Halaman (Harus jadi perintah streamlit pertama)
st.set_page_config(page_title="Analisis Sentimen Polri", layout="wide")
st.title("📊 Analisis Sentimen Polri (Lexicon + ML) — 2 Kelas")

# =====================================================================
# 1. PREPROCESSING & FILTER
# =====================================================================
def preprocess_text(text):
    """Membersihkan teks: hapus URL, mention, hashtag (#), karakter non-alfabet."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", " ", text) # Simpan kata dari hashtag
    text = re.sub(r"[^a-zA-Z\s]", "", text) # Hanya alfabet dan spasi
    text = re.sub(r"\s+", " ", text).strip() # Hapus spasi berlebih
    # Case folding (lowercase) akan dilakukan nanti
    return text

def is_relevant_to_polri(text_lower):
    """
    Mengecek relevansi teks (yang sudah lowercase) dengan keyword Polri
    dan mengecualikan keyword TNI.
    """
    # Menggunakan daftar keyword yang sudah diperluas
    keywords_polri = [
        # Institusi/Satuan Utama Polri
        "polri", "kepolisian", "mabes polri", "polda", "polres", "polsek", "polrestabes", "polresta",
        "brimob", "korbrimob", "gegana", "pelopor",
        "bareskrim", "ditreskrimum", "ditreskrimsus", "ditresnarkoba", # Direktorat Reserse
        "korlantas", "ditlantas", "satlantas", # Lalu Lintas
        "intelkam", "satintelkam", "densus", "densus 88", # Intelijen & Anti-Teror
        "propam", "divpropam", "paminal", "wabprof", "provos", # Pengawasan Internal
        "polairud", "korpolairud", # Polisi Air & Udara
        "sabhara", "samapta", "ditsamapta", "satsamapta", # Samapta/Patroli
        "binmas", "satbinmas", "bhabinkamtibmas", "polwan", # Polisi Wanita

        # Jabatan/Pangkat Umum Polri
        "polisi", "kapolri", "wakapolri", "kapolda", "wakapolda", "kapolres", "wakapolres", "kapolsek", "wakapolsek",
        "penyidik", "reskrim", "kasat", "kanit",
        "jenderal polisi", "komjen", "irjen", "brigjen", # Pati
        "kombes", "akbp", "kompol", # Pamen
        "akp", "iptu", "ipda", # Pama
        "aiptu", "aipda", "bripka", "brigpol", "brigadir", "briptu", "bripda", # Bintara
        "bharada", "bharatu", "bharaka" # Tamtama (umum + Brimob/Polairud)
    ]
    exclude_keywords = [
        # Institusi/Satuan Utama TNI
        "tni", "tentara", "angkatandarat", "angkatanlaut", "angkatanudara", "tni ad", "tni al", "tni au",
        "kodam", "korem", "kodim", "koramil", # Komando Wilayah AD
        "kostrad", "pangkostrad", "divif", # Komando Strategis AD
        "kopassus", "danjenkopassus", # Komando Pasukan Khusus AD
        "marinir", "kormar", "pasmar", # Korps Marinir AL
        "kopaska", "denjaka", # Pasukan Khusus AL
        "paskhas", "korpaskhas", "denbravo", # Pasukan Khas AU
        "armed", "kavaleri", "zeni", "arhanud", "yonif", # Beberapa kecabangan umum TNI AD

        # Jabatan/Pangkat Umum TNI
        "prajurit", "panglima tni", "ksad", "kasad", "ksal", "kasal", "ksau", "kasau", # Pimpinan & Jabatan Strategis
        "pangdam", "danrem", "dandim", "danramil", # Komandan Wilayah
        "jenderal tni", "laksamana", "marsekal", # Bintang 4
        "letjen", "laksdya", "marsdya", # Bintang 3
        "mayjen", "laksda", "marsda", # Bintang 2
        "brigjen tni", "laksma", "marsma", # Bintang 1
        "kolonel", "letkol", "mayor", # Pamen
        "kapten", "lettu", "letda", # Pama
        "peltu", "pelda", "serma", "serka", "sertu", "serda", # Bintara
        "kopka", "koptu", "kopda", "praka", "pratu", "prada" # Tamtama
    ]

    # Buat pola regex
    pattern_polri = r"\b(?:{})\b".format("|".join(keywords_polri))
    pattern_exclude = r"\b(?:{})\b".format("|".join(exclude_keywords))

    # Cek kecocokan (teks sudah diasumsikan lowercase)
    return bool(re.search(pattern_polri, text_lower)) and not re.search(pattern_exclude, text_lower)

# =====================================================================
# 2. LOAD LEXICON POSITIF & NEGATIF (GABUNGAN)
# =====================================================================
@st.cache_resource
def load_lexicons():
    """Memuat dan menggabungkan leksikon InSet (2 sumber) dan SentiStrength."""
    st.info("📚 Memuat kamus positif & negatif...")
    urls = {
        "fajri_pos": "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        "fajri_neg": "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv",
        "onpilot_pos": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv",
        "onpilot_neg": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv",
        "sentiwords_json": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt"
    }

    pos_lex = set()
    neg_lex = set()

    try:
        # Muat fajri91 (header=None, kolom 0)
        pos_lex.update(set(pd.read_csv(io.StringIO(requests.get(urls["fajri_pos"]).text), sep="\t", header=None, usecols=[0], names=['word'], on_bad_lines='skip', encoding='utf-8')['word'].dropna().astype(str)))
        neg_lex.update(set(pd.read_csv(io.StringIO(requests.get(urls["fajri_neg"]).text), sep="\t", header=None, usecols=[0], names=['word'], on_bad_lines='skip', encoding='utf-8')['word'].dropna().astype(str)))
        st.info("   -> OK: Leksikon fajri91 dimuat.")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat leksikon fajri91: {e}")

    try:
        # Muat onpilot (header=0, kolom 'word')
        pos_lex.update(set(pd.read_csv(io.StringIO(requests.get(urls["onpilot_pos"]).text), sep="\t", header=0, usecols=['word'], on_bad_lines='skip', encoding='utf-8')['word'].dropna().astype(str)))
        neg_lex.update(set(pd.read_csv(io.StringIO(requests.get(urls["onpilot_neg"]).text), sep="\t", header=0, usecols=['word'], on_bad_lines='skip', encoding='utf-8')['word'].dropna().astype(str)))
        st.info("   -> OK: Leksikon onpilot dimuat.")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat leksikon onpilot: {e}")

    # Muat SentiWords JSON
    try:
        senti_json = json.loads(requests.get(urls["sentiwords_json"]).text)
        for k, v in senti_json.items():
            if int(v) > 0:
                pos_lex.add(k)
            elif int(v) < 0:
                neg_lex.add(k)
        st.info(f"   -> OK: Leksikon SentiWords JSON dimuat ({len(senti_json)} entri).")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat sentiwords JSON: {e}")

    st.success(f"✅ Leksikon dimuat: {len(pos_lex)} kata positif unik, {len(neg_lex)} kata negatif unik.")
    return pos_lex, neg_lex

# Muat leksikon saat aplikasi dimulai
pos_lex, neg_lex = load_lexicons()

# =====================================================================
# 3. LABEL SENTIMEN (Hanya Positif & Negatif - Logika Disederhanakan)
# =====================================================================
def label_sentiment_two_class(text, pos_lex, neg_lex):
    """
    Memberi label Positif/Negatif berdasarkan jumlah kata.
    Default ke negatif jika skor 0 atau negatif.
    """
    if not isinstance(text, str) or not text.strip():
        return 'negatif' # Default jika input bukan string atau kosong

    tokens = text.split()
    pos = sum(1 for t in tokens if t in pos_lex)
    neg = sum(1 for t in tokens if t in neg_lex)

    # Logika 2 Kelas Sederhana:
    # Jika skor positif LEBIH BESAR dari negatif -> positif
    # Jika tidak (termasuk jika sama atau keduanya 0) -> negatif
    if pos > neg:
        return "positif"
    else:
        return "negatif"

# =====================================================================
# 4. PREPROCESS + FILTER + LABEL (Fungsi Terpadu)
# =====================================================================
@st.cache_data(show_spinner=False)
def preprocess_filter_label(_df, text_col, _pos_lex, _neg_lex):
    """Menerapkan cleaning, case folding, filter Polri, dan labeling lexicon 2 kelas."""
    st.info("Memulai preprocessing, filter, & pelabelan...")
    df_processed = _df.copy()
    total_awal = len(df_processed)
    progress_bar = st.progress(0, text="Memulai...")

    # Langkah 1: Preprocessing (Clean + Case Fold)
    progress_bar.progress(1/3, text="1/3 Preprocessing (Clean & Case Fold)...")
    df_processed['cleaned_text'] = df_processed[text_col].astype(str).progress_apply(preprocess_text)
    df_processed.dropna(subset=['cleaned_text'], inplace=True)
    df_processed = df_processed[df_processed['cleaned_text'].str.strip().astype(bool)]
    df_processed['case_folded_text'] = df_processed['cleaned_text'].str.lower()
    if df_processed.empty:
        st.warning("Tidak ada teks valid setelah cleaning.")
        progress_bar.empty()
        return pd.DataFrame(), total_awal, 0, 0

    # Langkah 2: Filter Polri
    progress_bar.progress(2/3, text="2/3 Memfilter data Polri...")
    df_filtered = df_processed[df_processed["case_folded_text"].progress_apply(is_relevant_to_polri)]
    total_filtered = len(df_filtered)
    if df_filtered.empty:
        st.warning("Tidak ada data relevan dengan Polri setelah filter.")
        progress_bar.empty()
        return pd.DataFrame(), total_awal, total_filtered, 0

    # Langkah 3: Labeling (pada case_folded_text yang sudah difilter)
    progress_bar.progress(3/3, text="3/3 Pelabelan Sentimen (Lexicon)...")
    # Buat salinan agar tidak mengubah slice
    df_labeled = df_filtered.copy()
    df_labeled["sentiment"] = df_labeled["case_folded_text"].progress_apply(
        lambda x: label_sentiment_two_class(x, _pos_lex, _neg_lex)
    )
    total_label = len(df_labeled)
    progress_bar.empty()

    st.success("Preprocessing, Filter, & Pelabelan Selesai.")
    # Kembalikan kolom yang relevan
    return df_labeled[["cleaned_text", "case_folded_text", "sentiment"]], total_awal, total_filtered, total_label

# =====================================================================
# 5. TRAIN MODEL + TF-IDF + EVALUASI
# =====================================================================
@st.cache_data(show_spinner=False) # Cache hasil model
def train_models(_df_processed, max_features=5000, test_size=0.3):
    """Melatih Naive Bayes dan SVM (Linear) menggunakan TF-IDF."""
    st.info(f"Memulai pelatihan model (Test Size: {test_size:.0%}, Max Features: {max_features})...")
    if _df_processed.empty or 'sentiment' not in _df_processed.columns:
         st.error("Data yang diproses kosong atau tidak memiliki kolom 'sentiment'.")
         return None
    if len(_df_processed['sentiment'].unique()) < 2:
         st.error("Hanya ditemukan 1 kelas sentimen. Tidak dapat melatih model.")
         return None

    # --- PERBAIKAN: Gunakan 'case_folded_text' untuk TF-IDF ---
    X, y = _df_processed["case_folded_text"], _df_processed["sentiment"]
    labels = sorted(y.unique())

    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        st.write(f"Data dibagi: {len(X_train)} train, {len(X_test)} test")
    except ValueError as e:
        st.error(f"Gagal membagi data (mungkin data terlalu sedikit): {e}")
        return None

    # TF-IDF
    st.write(f"Membuat fitur TF-IDF (max_features={max_features}, ngram=1-2)...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
    try:
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
    except Exception as e:
        st.error(f"Gagal saat TF-IDF: {e}"); return None

    results = {"labels": labels} # Simpan label untuk confusion matrix

    # Naive Bayes
    st.write("Melatih Naive Bayes...")
    nb = MultinomialNB(alpha=0.3) # Alpha dari kode Anda
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_pred)
    nb_report = classification_report(y_test, nb_pred, labels=labels, output_dict=True, zero_division=0)
    results['nb'] = {'acc': nb_acc, 'report': nb_report, 'model': nb, 'preds': nb_pred}
    st.write(f"Akurasi Naive Bayes: {nb_acc*100:.2f}%")

    # SVM (Linear Kernel)
    st.write("Melatih SVM (Linear)...")
    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_report = classification_report(y_test, svm_pred, labels=labels, output_dict=True, zero_division=0)
    results['svm'] = {'acc': svm_acc, 'report': svm_report, 'model': svm, 'preds': svm_pred}
    st.write(f"Akurasi SVM: {svm_acc*100:.2f}%")

    # Simpan data split dan vectorizer
    results.update({
        "vectorizer": vectorizer, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test
    })

    st.success("Pelatihan model ML selesai.")
    return results

# =====================================================================
# 6. VISUALISASI
# =====================================================================
def show_confusion(y_test, preds, model_name, labels):
    """Menampilkan confusion matrix."""
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4)) # Ukuran disesuaikan
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)

def show_wordcloud(_df):
    """Menampilkan word cloud untuk sentimen positif dan negatif."""
    st.header("🌈 Word Clouds Sentimen")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positif 😊")
        text_pos = " ".join(_df[_df["sentiment"] == "positif"]["case_folded_text"].values)
        if text_pos.strip():
            try:
                wc_pos = WordCloud(width=600, height=300, background_color="white", colormap="Greens").generate(text_pos)
                st.image(wc_pos.to_array(), use_column_width=True)
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
                st.image(wc_neg.to_array(), use_column_width=True)
            except Exception as e:
                st.warning(f"Gagal membuat word cloud negatif: {e}")
        else:
            st.write("Tidak ada data negatif untuk word cloud.")


def show_metric_comparison(nb_report, svm_report):
    """Menampilkan bar chart perbandingan metrik (weighted avg)."""
    metrics = ["precision", "recall", "f1-score"]
    values = {
        "Naive Bayes": [nb_report.get("weighted avg", {}).get(m, 0) for m in metrics],
        "SVM": [svm_report.get("weighted avg", {}).get(m, 0) for m in metrics]
    }
    df_metrics = pd.DataFrame(values, index=metrics)
    st.bar_chart(df_metrics)

# =====================================================================
# 7. UI: FILE CSV & TEKS TUNGGAL
# =====================================================================
# State management
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'ml_results' not in st.session_state: st.session_state.ml_results = None
if 'current_filename' not in st.session_state: st.session_state.current_filename = ""

tab1, tab2 = st.tabs(["📂 Analisis File CSV", "⌨️ Analisis Cepat Teks Tunggal"])

with tab1:
    st.header("Analisis Sentimen dari File CSV")
    uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"], label_visibility="collapsed")
    
    # Reset state jika file baru
    if uploaded and uploaded.name != st.session_state.current_filename:
        st.session_state.processed_df = None
        st.session_state.ml_results = None
        st.session_state.current_filename = uploaded.name
        if 'selectbox_column' in st.session_state: del st.session_state['selectbox_column']

    if uploaded:
        try:
            # Baca file
            with st.spinner(f"Membaca {uploaded.name}..."):
                try: df_input = pd.read_csv(uploaded)
                except UnicodeDecodeError: uploaded.seek(0); df_input = pd.read_csv(uploaded, encoding='latin1')
            st.success(f"File berhasil diunggah: {uploaded.name} ({len(df_input)} baris)")
            st.dataframe(df_input.head(), hide_index=True)

            # Pilih kolom
            available_columns = [""] + df_input.columns.tolist()
            col_index = 0
            if 'selectbox_column' in st.session_state and st.session_state.selectbox_column in available_columns:
                col_index = available_columns.index(st.session_state.selectbox_column)
            text_col = st.selectbox("Pilih Kolom Teks:", available_columns, index=col_index, key="selectbox_column")

            if text_col:
                st.info(f"Kolom teks yang dipilih: **{text_col}**")
                
                # Tombol Proses
                if st.button("🚀 1. Jalankan Preprocessing & Labeling Lexicon", key="btn_process"):
                    st.session_state.processed_df = None # Reset
                    st.session_state.ml_results = None
                    if text_col not in df_input.columns: st.error("Kolom teks tidak valid.")
                    elif df_input[text_col].isnull().all(): st.error(f"Kolom '{text_col}' kosong.")
                    else:
                        # Panggil fungsi preprocess + filter + label
                        df_processed, total_awal, total_filtered, total_label = preprocess_and_label(df_input, text_col, pos_lex, neg_lex)
                        st.session_state.processed_df = df_processed # Simpan hasil
                        
                        # Tampilkan metrik data
                        st.header("🧩 Hasil Preprocessing, Filter & Labeling")
                        colA, colB, colC = st.columns(3)
                        colA.metric("Total Data Awal", total_awal)
                        colB.metric("Data Setelah Filter Polri", total_filtered)
                        colC.metric("Data yang Dilabeli", total_label)

                # Tampilkan hasil labeling jika sudah ada
                if st.session_state.processed_df is not None:
                    if not st.session_state.processed_df.empty:
                        st.dataframe(st.session_state.processed_df.head(10), hide_index=True)
                        st.bar_chart(st.session_state.processed_df["sentiment"].value_counts())
                        
                        # Tampilkan Word Clouds
                        show_wordcloud(st.session_state.processed_df)

                        # Bagian ML
                        st.header("🤖 2. Pelatihan Model Machine Learning")
                        st.markdown("Model akan dilatih menggunakan **Label Lexicon** sebagai target dan **Teks (Case Folded)** sebagai fitur.")
                        
                        col_ml1, col_ml2 = st.columns(2)
                        with col_ml1:
                            test_size = st.slider("Pilih Test Size (Data Uji)", 0.1, 0.5, 0.3, step=0.05, format="%.0f%%", key="slider_test")
                        with col_ml2:
                            max_feat = st.slider("Max Features TF-IDF", 1000, 10000, 5000, step=1000, key="slider_maxfeat")

                        if st.button("🤖 Latih Model NB & SVM", key="btn_train"):
                            st.session_state.ml_results = None # Reset
                            results = train_models(st.session_state.processed_df, max_feat, test_size)
                            st.session_state.ml_results = results # Simpan hasil

                    else:
                        st.warning("⚠️ Tidak ada data relevan dengan Polri setelah filter.")

                # Tampilkan hasil ML jika ada
                if st.session_state.ml_results is not None:
                    results = st.session_state.ml_results
                    st.header("📈 3. Hasil & Evaluasi Model")
                    
                    colD, colE = st.columns(2)
                    colD.metric("Akurasi Naive Bayes", f"{results['nb']['acc']:.2%}")
                    colE.metric("Akurasi SVM (Linear)", f"{results['svm']['acc']:.2%}")

                    st.subheader("Perbandingan Metrik (Weighted Avg)")
                    show_metric_comparison(results["nb_report"], results["svm_report"])
                    
                    st.subheader("Confusion Matrix")
                    colF, colG = st.columns(2)
                    with colF:
                        show_confusion(results["y_test"], results["nb_pred"], "Naive Bayes", results["labels"])
                    with colG:
                        show_confusion(results["y_test"], results["svm_pred"], "SVM (Linear)", results["labels"])
                    
                    st.subheader("Laporan Klasifikasi Detail")
                    with st.expander("Lihat Laporan Naive Bayes"):
                        st.dataframe(pd.DataFrame(results['nb_report']).transpose())
                    with st.expander("Lihat Laporan SVM (Linear)"):
                        st.dataframe(pd.DataFrame(results['svm_report']).transpose())

                    # Download button
                    st.header("📥 4. Unduh Hasil Labeling")
                    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
                        csv_data = st.session_state.processed_df.to_csv(index=False).encode("utf-8")
                        base_filename = st.session_state.current_filename.split('.')[0] if '.' in st.session_state.current_filename else st.session_state.current_filename
                        st.download_button(
                            "Unduh CSV Hasil Labeling",
                            csv_data,
                            f"hasil_sentimen_polri_{base_filename}.csv",
                            "text/csv",
                            key="download_csv_final"
                        )
                    
            elif not text_col:
                st.warning("☝️ Pilih kolom teks untuk memulai proses.")
                
        except pd.errors.EmptyDataError:
            st.error("File CSV yang diunggah kosong.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.exception(e) # Tampilkan traceback untuk debug

    else:
        st.info("Silakan unggah file CSV untuk memulai analisis.")

# ==============================================================================
# 🟩 TAB 2: INPUT TEKS (Fungsi 'analyze_single_text' sudah diperbarui)
# ==============================================================================
with tab2:
    st.header("💬 Analisis Cepat Teks Tunggal")
    input_text = st.text_area("Ketik atau paste teks di sini:", height=150, key="text_area_single")

    if st.button("🔍 Analisis Teks Ini", key="button_analyze_single"):
        if input_text and input_text.strip():
            with st.spinner("Menganalisis teks..."):
                # Panggil analyze_single_text (yang sudah ada filter)
                sentiment, cleaned = analyze_single_text(input_text, pos_lex, neg_lex)

            st.subheader("Hasil Analisis:")
            st.write("**Teks Setelah Preprocessing:**")
            st.info(f"`{cleaned}`") # Tampilkan teks cleaned
            st.write("**Hasil Sentimen:**")

            if sentiment == "positif":
                st.success("✅ Sentimen: POSITIF 😊")
            elif sentiment == "negatif":
                st.error("❌ Sentimen: NEGATIF 😠")
            elif sentiment == "tidak relevan":
                 st.warning("⚠️ Sentimen: TIDAK RELEVAN (Tidak terdeteksi keyword Polri atau terdeteksi keyword TNI).")
            else: # 'tidak valid'
                 st.warning("⚠️ Teks tidak valid atau menjadi kosong setelah preprocessing.")

        else:
            st.warning("Masukkan teks terlebih dahulu sebelum menganalisis.")

# --- Footer ---
st.markdown("---")
st.markdown("Aplikasi Analisis Sentimen Polri")
