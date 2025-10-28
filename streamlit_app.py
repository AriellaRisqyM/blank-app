# =====================================================================
# STREAMLIT: Analisis Sentimen Polri (Lexicon + ML) — FINAL SKRIPSI
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
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def is_relevant_to_polri(text):
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
# 🔹 2. LOAD KAMUS & LEXICON (InSet + onpilot + Sentistrength)
# =====================================================================
@st.cache_resource
def load_lexicons():
    st.info("📚 Memuat kamus dan leksikon sentimen dari semua sumber...")

    urls = {
        "normalisasi": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt",
        "stopword": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt",
        "fajri_pos": "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv",
        "fajri_neg": "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv",
        "onpilot_pos": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv",
        "onpilot_neg": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv",
        "sentiwords_json": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/_json_sentiwords_id.txt",
        "boosterwords": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/boosterwords_id.txt",
        "emoticon": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/emoticon_id.txt",
        "idioms": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/idioms_id.txt",
        "negating": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/negatingword.txt",
        "question": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/questionword.txt",
        "sentiwords_txt": "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/sentistrength_id/sentiwords_id.txt"
    }

    kamus_normalisasi = json.loads(requests.get(urls["normalisasi"]).text)
    stopwords = set(requests.get(urls["stopword"]).text.splitlines())

    # Lexicon gabungan InSet + onpilot
    pos_lex = set(pd.read_csv(urls["fajri_pos"], sep="\t", header=None)[0].dropna().astype(str)) \
              .union(set(pd.read_csv(urls["onpilot_pos"], sep="\t", header=None)[0].dropna().astype(str)))
    neg_lex = set(pd.read_csv(urls["fajri_neg"], sep="\t", header=None)[0].dropna().astype(str)) \
              .union(set(pd.read_csv(urls["onpilot_neg"], sep="\t", header=None)[0].dropna().astype(str)))

    # Sentistrength JSON lexicon
    try:
        senti_json = json.loads(requests.get(urls["sentiwords_json"]).text)
        for k, v in senti_json.items():
            if int(v) > 0:
                pos_lex.add(k)
            elif int(v) < 0:
                neg_lex.add(k)
    except:
        st.warning("⚠️ Gagal memuat sentiwords JSON")

    # TXT lexicons: booster, emoticon, idioms, negating, question, sentiwords
    for key in ["boosterwords", "emoticon", "idioms", "negating", "question", "sentiwords_txt"]:
        try:
            words = set(requests.get(urls[key]).text.splitlines())
            pos_lex.update(words)
        except:
            st.warning(f"⚠️ Gagal memuat {key}")

    st.success(f"✅ Leksikon dimuat: {len(pos_lex)} positif, {len(neg_lex)} negatif, {len(stopwords)} stopword.")
    return kamus_normalisasi, stopwords, pos_lex, neg_lex

kamus_normalisasi, stopword_set, pos_lex, neg_lex = load_lexicons()

# =====================================================================
# 🔹 3. LABELING SENTIMEN
# =====================================================================
def label_sentiment_two_class(text, pos_lex, neg_lex):
    tokens = [t for t in text.split() if t not in stopword_set]
    pos = sum(1 for t in tokens if t in pos_lex)
    neg = sum(1 for t in tokens if t in neg_lex)
    if pos == 0 and neg == 0:
        return "netral"
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
    svm_pred =
