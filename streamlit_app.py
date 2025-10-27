# ===============================================================
# app_sentimen_polri_final_modular.py
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import re, requests, json, io, nltk
from tqdm.auto import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

tqdm.pandas()
st.set_page_config(layout="wide", page_title="Analisis Sentimen Polri Modular")

# ===============================================================
# 1️⃣ LOAD RESOURCE (KAMUS & LEXICON)
# ===============================================================
@st.cache_resource
def load_resources():
    # Kamus normalisasi & stopword
    norm_url = "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt"
    stop_url = "https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt"
    norm = requests.get(norm_url).json()
    stop = set(requests.get(stop_url).text.splitlines())

    # Leksikon InSet
    pos1 = pd.read_csv("https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv", sep="\t", header=None)[0].tolist()
    neg1 = pd.read_csv("https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv", sep="\t", header=None)[0].tolist()
    pos2 = pd.read_csv("https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv", sep="\t", header=0)["word"].tolist()
    neg2 = pd.read_csv("https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv", sep="\t", header=0)["word"].tolist()
    pos_lex = set(pos1 + pos2)
    neg_lex = set(neg1 + neg2)

    stemmer = StemmerFactory().create_stemmer()
    return norm, stop, stemmer, pos_lex, neg_lex

norm_dict, stop_words, stemmer, pos_lex, neg_lex = load_resources()

# ===============================================================
# 2️⃣ PREPROCESSING FUNCTIONS
# ===============================================================
def clean_text(t):
    if not isinstance(t, str): return ""
    t = re.sub(r"http\S+|@\w+|#"," ",t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()

def tokenize(text): return text.split()
def normalize_token(tokens): return [norm_dict.get(w, w) for w in tokens]
def remove_stop(tokens): return [w for w in tokens if w not in stop_words and len(w)>1]
def stem_tokens(tokens): return [stemmer.stem(w) for w in tokens]

# ===============================================================
# 3️⃣ FILTER POLRI (HANYA JABATAN & LEMBAGA)
# ===============================================================
def is_relevant_to_polri(text):
    keywords_polri = [
        'polri','polisi','kepolisian','polda','polres','polsek','polresta','polrestabes',
        'poltabes','kapolri','wakapolri','kapolda','wakapolda','kapolres','wakapolres',
        'kapolsek','wakapolsek','kabid humas','kadiv humas','kasatlantas','kasatreskrim',
        'kasatresnarkoba','kasatintelkam','kasatbinmas','brimob','sabhara','lantas',
        'reskrim','intelkam','tipikor','tipidkor','tipidnarkoba','tipidter','densus',
        'spkt','bareksrim','bhabinkamtibmas','polair','polairud','paminal','propam',
        'polantas','bhayangkara','polwan','polsus'
    ]
    pattern_polri = r'\b(?:'+'|'.join(keywords_polri)+r')\b'
    return bool(re.search(pattern_polri, text))

# ===============================================================
# 4️⃣ LABELING (2 KELAS)
# ===============================================================
def label_sentiment(tokens):
    if not isinstance(tokens, list): return "negatif"
    pos = sum(1 for w in tokens if w in pos_lex)
    neg = sum(1 for w in tokens if w in neg_lex)
    return "positif" if pos >= neg else "negatif"

# ===============================================================
# 5️⃣ MACHINE LEARNING
# ===============================================================
def train_models(df):
    X, y = df["case_folded"], df["sentiment"]
    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    tfidf = TfidfVectorizer(max_features=1500)
    Xtr_tf, Xts_tf = tfidf.fit_transform(Xtr), tfidf.transform(Xts)

    nb = MultinomialNB(alpha=0.2).fit(Xtr_tf, ytr)
    svm = SVC(kernel='linear', probability=True).fit(Xtr_tf, ytr)

    preds_nb, preds_svm = nb.predict(Xts_tf), svm.predict(Xts_tf)
    prob_nb, prob_svm = nb.predict_proba(Xts_tf)[:,1], svm.predict_proba(Xts_tf)[:,1]

    metrics = {}
    for name, pred, prob, model in [
        ("Naive Bayes", preds_nb, prob_nb, nb),
        ("SVM Linear", preds_svm, prob_svm, svm)
    ]:
        acc = accuracy_score(yts, pred)
        prec = precision_score(yts, pred, pos_label="positif")
        rec = recall_score(yts, pred, pos_label="positif")
        f1 = f1_score(yts, pred, pos_label="positif")
        fpr, tpr, _ = roc_curve(yts.map({"negatif":0,"positif":1}), prob)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(yts, pred, labels=["positif","negatif"])
        metrics[name] = {"acc":acc,"prec":prec,"rec":rec,"f1":f1,"roc":(fpr,tpr,roc_auc),"cm":cm}
    return metrics, tfidf.get_feature_names_out(), Xtr_tf

# ===============================================================
# 6️⃣ VISUALISASI
# ===============================================================
def plot_cm(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax,
                xticklabels=["Positif","Negatif"], yticklabels=["Positif","Negatif"])
    ax.set_title(title)
    st.pyplot(fig)

def plot_roc(metrics):
    fig, ax = plt.subplots()
    for name, v in metrics.items():
        fpr, tpr, roc_auc = v["roc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); st.pyplot(fig)

def show_wordcloud(df, label):
    text = " ".join(df[df["sentiment"]==label]["case_folded"])
    if text.strip():
        wc = WordCloud(width=800,height=400,background_color='white',
                       colormap='Greens' if label=="positif" else 'Reds').generate(text)
        st.image(wc.to_array(), caption=f"WordCloud - {label}")

# ===============================================================
# 7️⃣ STREAMLIT PIPELINE
# ===============================================================
st.title("📊 Analisis Sentimen Polri Modular (Lexicon + ML)")

# SESSION STATE
for k in ["df_raw","df_pre","df_filt","df_label","tfidf","terms","Xtf","metrics"]:
    if k not in st.session_state: st.session_state[k] = None

# STEP 1: UPLOAD
st.header("1️⃣ Upload Dataset")
upl = st.file_uploader("Unggah CSV/Excel", type=["csv","xlsx"])
if upl:
    df = pd.read_csv(upl) if upl.name.endswith("csv") else pd.read_excel(upl)
    st.session_state.df_raw = df
    st.success(f"✅ File '{upl.name}' dimuat ({len(df)} baris).")
    st.dataframe(df.head())

# STEP 2: PREPROCESSING
if st.session_state.df_raw is not None:
    st.header("2️⃣ Preprocessing")
    text_col = st.selectbox("Pilih kolom teks:", st.session_state.df_raw.columns)
    if st.button("🧹 Jalankan Preprocessing"):
        df = st.session_state.df_raw.copy()
        df["cleaned"] = df[text_col].astype(str).apply(clean_text)
        df["case_folded"] = df["cleaned"].str.lower()
        st.session_state.df_pre = df
        st.success("✅ Preprocessing selesai.")
        st.dataframe(df[[text_col,"cleaned","case_folded"]].head(10))

# STEP 3: FILTER POLRI
if st.session_state.df_pre is not None:
    st.header("3️⃣ Filter Polri")
    if st.button("🚓 Jalankan Filter Polri"):
        df = st.session_state.df_pre.copy()
        df = df[df["case_folded"].apply(is_relevant_to_polri)]
        st.session_state.df_filt = df
        st.info(f"Jumlah data relevan Polri: {len(df)}")
        st.dataframe(df.head(10))

# STEP 4: LABELING
if st.session_state.df_filt is not None:
    st.header("4️⃣ Pelabelan Lexicon (2 kelas)")
    if st.button("🏷️ Jalankan Pelabelan"):
        df = st.session_state.df_filt.copy()
        df["tokens"] = df["case_folded"].apply(tokenize)
        df["normalized"] = df["tokens"].apply(normalize_token)
        df["no_stop"] = df["normalized"].apply(remove_stop)
        df["stemmed"] = df["no_stop"].apply(stem_tokens)
        df["sentiment"] = df["stemmed"].apply(label_sentiment)
        st.session_state.df_label = df
        st.success("✅ Pelabelan selesai.")
        st.bar_chart(df["sentiment"].value_counts())
        st.dataframe(df[["case_folded","sentiment"]].head(10))

# STEP 5: TF-IDF + MODEL
if st.session_state.df_label is not None:
    st.header("5️⃣ TF-IDF & Latih Model")
    if st.button("🤖 Jalankan Model ML"):
        df = st.session_state.df_label.copy()
        metrics, terms, Xtf = train_models(df)
        st.session_state.metrics, st.session_state.terms, st.session_state.Xtf = metrics, terms, Xtf
        st.success("✅ Pelatihan Model Selesai.")
        eval_table = pd.DataFrame({
            "Model":[m for m in metrics],
            "Akurasi":[f"{v['acc']:.3f}" for v in metrics.values()],
            "Presisi":[f"{v['prec']:.3f}" for v in metrics.values()],
            "Recall":[f"{v['rec']:.3f}" for v in metrics.values()],
            "F1":[f"{v['f1']:.3f}" for v in metrics.values()]
        })
        st.dataframe(eval_table)

# STEP 6: VISUALISASI
if st.session_state.metrics is not None:
    st.header("6️⃣ Evaluasi & Visualisasi")
    metrics = st.session_state.metrics

    st.subheader("📉 Confusion Matrix")
    for m,v in metrics.items(): plot_cm(v["cm"],m)

    st.subheader("📈 ROC Curve")
    plot_roc(metrics)

    st.subheader("📊 Top 10 Term Frequency")
    Xtf, terms = st.session_state.Xtf, st.session_state.terms
    tf_counts = np.array(Xtf.sum(axis=0)).flatten()
    freq_df = pd.DataFrame({"term":terms,"freq":tf_counts}).sort_values("freq",ascending=False).head(10)
    st.dataframe(freq_df)
    fig, ax = plt.subplots(); ax.barh(freq_df["term"],freq_df["freq"]); ax.invert_yaxis()
    st.pyplot(fig)

    st.subheader("☁️ WordCloud")
    col1, col2 = st.columns(2)
    with col1: show_wordcloud(st.session_state.df_label,"positif")
    with col2: show_wordcloud(st.session_state.df_label,"negatif")

    st.download_button("📥 Unduh Hasil CSV",
        st.session_state.df_label.to_csv(index=False).encode('utf-8'),
        file_name="hasil_sentimen_polri.csv")
