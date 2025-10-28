# =====================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML) — Versi Modular
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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from imblearn.over_sampling import SMOTE

tqdm.pandas()
st.set_page_config(page_title="Analisis Sentimen Polri Modular", layout="wide")
st.title("📊 Analisis Sentimen Polri (Lexicon + Machine Learning — Step by Step)")

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

def normalize_and_stem(text, kamus_normalisasi):
    """Normalisasi slang dan stemming bahasa Indonesia."""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = text.split()
    normalized = []
    for t in tokens:
        t = kamus_normalisasi.get(t, t)
        normalized.append(stemmer.stem(t))
    return " ".join(normalized)

# ⚠️ Fungsi relevansi TIDAK DIUBAH
def is_relevant_to_polri(text):
    """Cek relevansi teks terhadap Polri."""
    keywords_polri = [
        "polri", "kepolisian", "mabes polri", "polda", "polres", "polsek", "polrestabes", "polresta",
        "brimob", "korbrimob", "gegana", "pelopor",
        "bareskrim", "ditreskrimum", "ditreskrimsus", "ditresnarkoba",
        "korlantas", "ditlantas", "satlantas",
        "intelkam", "satintelkam", "densus", "densus 88",
        "propam", "divpropam", "paminal", "wabprof", "provos",
        "polairud", "korpolairud",
        "sabhara", "samapta", "ditsamapta", "satsamapta",
        "binmas", "satbinmas", "bhabinkamtibmas",
        "polwan",
        "polisi", "kapolri", "wakapolri", "kapolda", "wakapolda", "kapolres", "wakapolres",
        "kapolsek", "wakapolsek", "penyidik", "reskrim", "kasat", "kanit",
        "jenderal polisi", "komjen", "irjen", "brigjen",
        "kombes", "akbp", "kompol",
        "akp", "iptu", "ipda",
        "aiptu", "aipda", "bripka", "brigpol", "brigadir", "briptu", "bripda",
        "bharada", "bharatu", "bharaka"
    ]
    exclude_keywords = [
        "tni", "tentara", "angkatandarat", "angkatanlaut", "angkatanudara", "tni ad", "tni al", "tni au",
        "kodam", "korem", "kodim", "koramil",
        "kostrad", "pangkostrad", "divif",
        "kopassus", "danjenkopassus",
        "marinir", "kormar", "pasmar",
        "kopaska", "denjaka",
        "paskhas", "korpaskhas", "denbravo",
        "armed", "kavaleri", "zeni", "arhanud", "yonif",
        "prajurit", "panglima tni", "ksad", "kasad", "ksal", "kasal", "ksau", "kasau",
        "pangdam", "danrem", "dandim", "danramil",
        "jenderal tni", "laksamana", "marsekal",
        "letjen", "laksdya", "marsdya",
        "mayjen", "laksda", "marsda",
        "brigjen tni", "laksma", "marsma",
        "kolonel", "letkol", "mayor",
        "kapten", "lettu", "letda",
        "peltu", "pelda", "serma", "serka", "sertu", "serda",
        "kopka", "koptu", "kopda", "praka", "pratu", "prada"
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
# 🔹 4. VISUALISASI
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
# 🔹 5. UI STREAMLIT MODULAR
# =====================================================================
st.sidebar.header("⚙️ Pengaturan Analisis")
test_size = st.sidebar.slider("Proporsi Data Uji", 0.1, 0.5, 0.3, step=0.05)
max_feat = st.sidebar.slider("Max Features TF-IDF", 1000, 10000, 5000, step=1000)

tab1, tab2 = st.tabs(["📂 Analisis File CSV", "⌨️ Analisis Cepat Teks Tunggal"])

# =================== TAB 1 ===================
with tab1:
    uploaded = st.file_uploader("Unggah Dataset CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Pilih Kolom Teks", df.columns.tolist())

        # --- Step 1: PREPROCESSING ---
        if st.button("🧩 1️⃣ Jalankan Preprocessing & Filter"):
            df[text_col] = df[text_col].astype(str)
            df["cleaned_text"] = df[text_col].apply(preprocess_text)
            df["cleaned_text"] = df["cleaned_text"].apply(lambda x: normalize_and_stem(x, kamus_normalisasi))
            df_filtered = df[df["cleaned_text"].apply(is_relevant_to_polri)]
            st.session_state["df_filtered"] = df_filtered
            st.success(f"✅ Selesai! Total Awal: {len(df)}, Setelah Filter Polri: {len(df_filtered)}")

        # --- Step 2: LABELING ---
        if "df_filtered" in st.session_state and st.button("🏷️ 2️⃣ Jalankan Pelabelan Sentimen"):
            df_filtered = st.session_state["df_filtered"].copy()
            df_filtered["sentiment"] = df_filtered["cleaned_text"].progress_apply(
                lambda x: label_sentiment_two_class(x, pos_lex, neg_lex)
            )
            st.session_state["df_labeled"] = df_filtered
            st.success(f"✅ Pelabelan selesai! Total Data: {len(df_filtered)}")
            st.dataframe(df_filtered.head(10))
            st.bar_chart(df_filtered["sentiment"].value_counts())

        # --- Step 3: TF-IDF + MODEL ---
        if "df_labeled" in st.session_state and st.button("🧠 3️⃣ Jalankan TF-IDF & Latih Model"):
            df_labeled = st.session_state["df_labeled"]
            X, y = df_labeled["cleaned_text"], df_labeled["sentiment"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            vectorizer = TfidfVectorizer(max_features=max_feat, ngram_range=(1,3), sublinear_tf=True)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            sm = SMOTE(random_state=42)
            X_train_tfidf, y_train = sm.fit_resample(X_train_tfidf, y_train)

            nb = MultinomialNB(alpha=0.2)
            nb.fit(X_train_tfidf, y_train)
            nb_pred = nb.predict(X_test_tfidf)

            svm = SVC(kernel="linear", C=2, class_weight="balanced", probability=True, random_state=42)
            svm.fit(X_train_tfidf, y_train)
            svm_pred = svm.predict(X_test_tfidf)

            st.session_state["model_results"] = {
                "y_test": y_test, "nb_pred": nb_pred, "svm_pred": svm_pred,
                "nb_report": classification_report(y_test, nb_pred, output_dict=True),
                "svm_report": classification_report(y_test, svm_pred, output_dict=True)
            }
            st.success("✅ Model Naive Bayes & SVM selesai dilatih!")

        # --- Step 4: EVALUASI ---
        if "model_results" in st.session_state and st.button("📈 4️⃣ Tampilkan Evaluasi Model"):
            results = st.session_state["model_results"]
            y_test, nb_pred, svm_pred = results["y_test"], results["nb_pred"], results["svm_pred"]
            nb_acc = accuracy_score(y_test, nb_pred)
            svm_acc = accuracy_score(y_test, svm_pred)

            col1, col2 = st.columns(2)
            col1.metric("Akurasi Naive Bayes", f"{nb_acc*100:.2f}%")
            col2.metric("Akurasi SVM", f"{svm_acc*100:.2f}%")

            col3, col4 = st.columns(2)
            with col3:
                show_confusion(y_test, nb_pred, "Naive Bayes")
            with col4:
                show_confusion(y_test, svm_pred, "SVM")

            st.subheader("📋 Laporan Evaluasi Model")
            st.write("**Naive Bayes**")
            st.dataframe(pd.DataFrame(results["nb_report"]).transpose())
            st.write("**SVM**")
            st.dataframe(pd.DataFrame(results["svm_report"]).transpose())

            st.subheader("🌈 WordCloud Sentimen")
            show_wordcloud(st.session_state["df_labeled"])

# =================== TAB 2 ===================
with tab2:
    st.subheader("💬 Analisis Cepat Teks Tunggal")
    input_text = st.text_area("Ketik atau paste teks di sini:", height=150)
    if st.button("🔍 Analisis Teks Ini"):
        if input_text.strip():
            cleaned = preprocess_text(input_text)
            cleaned = normalize_and_stem(cleaned, kamus_normalisasi)
            sentiment = label_sentiment_two_class(cleaned, pos_lex, neg_lex)
            st.info(f"Teks setelah preprocessing:\n{cleaned}")
            if sentiment == "positif":
                st.success("✅ Sentimen: POSITIF 😊")
            else:
                st.error("❌ Sentimen: NEGATIF 😠")
        else:
            st.warning("Masukkan teks terlebih dahulu.")
