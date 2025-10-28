# =====================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML)
# =====================================================================
import streamlit as st
import pandas as pd
import requests, re, json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

tqdm.pandas()
st.set_page_config(page_title="Analisis Sentimen Polri", layout="wide")
st.title("📊 Analisis Sentimen Polri (Lexicon + Machine Learning)")

# =====================================================================
# 🔹 1. PREPROCESSING
# =====================================================================
def preprocess_text(text):
    """Membersihkan teks dari URL, mention, hashtag, simbol."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def is_relevant_to_polri(text):
    """Cek relevansi teks terhadap Polri."""
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
        "binmas", "satbinmas", "bhabinkamtibmas", # Pembinaan Masyarakat
        "polwan", # Polisi Wanita
    
        # Jabatan/Pangkat Umum Polri
        "polisi", "kapolri", "wakapolri", "kapolda", "wakapolda", "kapolres", "wakapolres",
        "kapolsek", "wakapolsek", "penyidik", "reskrim", "kasat", "kanit",
        "jenderal polisi", "komjen", "irjen", "brigjen", # Pati
        "kombes", "akbp", "kompol", # Pamen
        "akp", "iptu", "ipda", # Pama
        "aiptu", "aipda", "bripka", "brigpol", "brigadir", "briptu", "bripda", # Bintara
        "bharada", "bharatu", "bharaka" # Tamtama (umum + Brimob/Polairud)
    ]

    exclude_keywords = [
        # Institusi/Satuan Utama TNI
        "tni"
        
    ]

    pattern_polri = r"\b(?:{})\b".format("|".join(keywords_polri))
    pattern_exclude = r"\b(?:{})\b".format("|".join(exclude_keywords))
    return bool(re.search(pattern_polri, text)) and not re.search(pattern_exclude, text)

# =====================================================================
# 🔹 2. LOAD KAMUS & LEXICON
# =====================================================================
@st.cache_resource
def load_lexicons():
    st.info("📚 Memuat kamus dan leksikon sentimen...")
    urls = {
        "normalisasi": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt",
        "stopword": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt",
        "inset_pos": "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        "inset_neg": "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
    }
    kamus_normalisasi = json.loads(requests.get(urls["normalisasi"]).text)
    stopwords = set(requests.get(urls["stopword"]).text.splitlines())
    pos = set(pd.read_csv(urls["inset_pos"], sep="\t", header=None)[0].dropna().astype(str))
    neg = set(pd.read_csv(urls["inset_neg"], sep="\t", header=None)[0].dropna().astype(str))
    st.success(f"✅ Leksikon dimuat: {len(pos)} positif, {len(neg)} negatif, {len(stopwords)} stopword.")
    return kamus_normalisasi, stopwords, pos, neg

kamus_normalisasi, stopword_set, pos_lex, neg_lex = load_lexicons()

# =====================================================================
# 🔹 3. LABELING SENTIMEN
# =====================================================================
def label_sentiment_two_class(text, pos_lex, neg_lex):
    tokens = [t for t in text.split() if t not in stopword_set]
    pos = sum(1 for t in tokens if t in pos_lex)
    neg = sum(1 for t in tokens if t in neg_lex)
    if pos == 0 and neg == 0:
        return "negatif"
    return "positif" if pos >= neg else "negatif"

# =====================================================================
# 🔹 4. PREPROCESS + LABEL + FILTER
# =====================================================================
@st.cache_data
def preprocess_and_label(df, text_col, pos_lex, neg_lex):
    total_awal = len(df)
    df[text_col] = df[text_col].astype(str)
    df["cleaned_text"] = df[text_col].apply(preprocess_text)
    df_filtered = df[df["cleaned_text"].apply(is_relevant_to_polri)]
    total_filtered = len(df_filtered)

    if df_filtered.empty:
        return pd.DataFrame(), total_awal, total_filtered, 0

    df_filtered["sentiment"] = df_filtered["cleaned_text"].progress_apply(
        lambda x: label_sentiment_two_class(x, pos_lex, neg_lex)
    )
    total_label = len(df_filtered)
    return df_filtered[["cleaned_text", "sentiment"]], total_awal, total_filtered, total_label

# =====================================================================
# 🔹 5. TRAIN MODEL + TF-IDF + EVALUASI
# =====================================================================
def train_models(df, max_features=5000, test_size=0.3):
    X, y = df["cleaned_text"], df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    nb = MultinomialNB(alpha=0.3)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)

    svm = SVC(kernel="linear", probability=True, random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)

    nb_report = classification_report(y_test, nb_pred, output_dict=True)
    svm_report = classification_report(y_test, svm_pred, output_dict=True)

    return {
        "vectorizer": vectorizer, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "nb": nb, "svm": svm,
        "nb_pred": nb_pred, "svm_pred": svm_pred,
        "nb_report": nb_report, "svm_report": svm_report
    }

# =====================================================================
# 🔹 6. VISUALISASI & EVALUASI
# =====================================================================
def show_confusion(y_test, preds, model_name):
    cm = confusion_matrix(y_test, preds, labels=["positif", "negatif"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positif", "Negatif"],
                yticklabels=["Positif", "Negatif"])
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)

def show_wordcloud(df):
    for label, color in [("positif", "Greens"), ("negatif", "Reds")]:
        text = " ".join(df[df["sentiment"] == label]["cleaned_text"].values)
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=color, stopwords=stopword_set).generate(text)
        st.image(wc.to_array(), caption=f"WordCloud - {label.capitalize()}")

# =====================================================================
# 🔹 7. UI: TAB FILE & INPUT TEKS
# =====================================================================
tab1, tab2 = st.tabs(["📂 Analisis File CSV", "⌨️ Analisis Cepat Teks Tunggal"])

# =================== TAB 1 ===================
with tab1:
    uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"])
    test_size = st.slider("Pilih Test Size (Data Uji)", 0.1, 0.5, 0.3, step=0.05)
    max_feat = st.slider("Max Features TF-IDF", 1000, 10000, 5000, step=1000)

    if uploaded:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Pilih Kolom Teks", df.columns.tolist())

        if st.button("🚀 Jalankan Analisis File"):
            st.header("🧩 Tahap 1: Preprocessing & Labeling")
            df_processed, total_awal, total_filtered, total_label = preprocess_and_label(df, text_col, pos_lex, neg_lex)

            colA, colB, colC = st.columns(3)
            colA.metric("Total Data Awal", total_awal)
            colB.metric("Data Setelah Filter Polri", total_filtered)
            colC.metric("Data yang Dilabeli", total_label)

            if not df_processed.empty:
                st.dataframe(df_processed.head(10))
                st.bar_chart(df_processed["sentiment"].value_counts())

                st.header("🧠 Tahap 2: TF-IDF dan Pembagian Data")
                results = train_models(df_processed, max_feat, test_size)
                st.success("✅ TF-IDF dan split data berhasil!")

                colD, colE = st.columns(2)
                colD.metric("Data Latih", len(results["X_train"]))
                colE.metric("Data Uji", len(results["X_test"]))

                st.header("🌈 Tahap 3: WordCloud Sentimen")
                show_wordcloud(df_processed)

                st.header("🤖 Tahap 4: Pelatihan Model Naive Bayes & SVM")
                nb_acc = accuracy_score(results["y_test"], results["nb_pred"])
                svm_acc = accuracy_score(results["y_test"], results["svm_pred"])
                st.metric("Akurasi Naive Bayes", f"{nb_acc*100:.2f}%")
                st.metric("Akurasi SVM", f"{svm_acc*100:.2f}%")

                col1, col2 = st.columns(2)
                with col1:
                    show_confusion(results["y_test"], results["nb_pred"], "Naive Bayes")
                with col2:
                    show_confusion(results["y_test"], results["svm_pred"], "SVM")

                st.header("📈 Tahap 5: Evaluasi Model")
                st.subheader("Naive Bayes")
                st.dataframe(pd.DataFrame(results["nb_report"]).transpose())
                st.subheader("SVM")
                st.dataframe(pd.DataFrame(results["svm_report"]).transpose())

                st.download_button(
                    "📥 Unduh Hasil Labeling CSV",
                    df_processed.to_csv(index=False).encode("utf-8"),
                    "hasil_sentimen_polri.csv",
                    "text/csv"
                )
            else:
                st.warning("⚠️ Tidak ada data relevan dengan Polri setelah filter.")

# =================== TAB 2 ===================
with tab2:
    st.subheader("💬 Analisis Cepat Teks Tunggal")
    input_text = st.text_area("Ketik atau paste teks di sini:", height=150)
    if st.button("🔍 Analisis Teks Ini"):
        if input_text.strip():
            cleaned = preprocess_text(input_text)
            if not is_relevant_to_polri(cleaned):
                st.warning("⚠️ Teks tidak relevan dengan Polri.")
            else:
                sentiment = label_sentiment_two_class(cleaned, pos_lex, neg_lex)
                st.info(f"Teks setelah preprocessing:\n{cleaned}")
                if sentiment == "positif":
                    st.success("✅ Sentimen: POSITIF 😊")
                else:
                    st.error("❌ Sentimen: NEGATIF 😠")
        else:
            st.warning("Masukkan teks terlebih dahulu.")
