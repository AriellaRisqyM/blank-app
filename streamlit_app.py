# ===============================================================
# app_sentimen_polri_final.py
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
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

tqdm.pandas()
st.set_page_config(layout="wide", page_title="Analisis Sentimen Polri (Lexicon + ML)")

# ===============================================================
# 1️⃣  KAMUS & LEXICON LOADING
# ===============================================================
@st.cache_resource
def load_json(url):
    r = requests.get(url); r.raise_for_status(); return r.json()

@st.cache_resource
def load_text(url):
    r = requests.get(url); r.raise_for_status(); return r.text.splitlines()

@st.cache_resource
def load_lexicon(url, header=None):
    df = pd.read_csv(url, sep="\t", header=header, names=["word"], usecols=[0])
    return set(df["word"].dropna().astype(str).unique())

@st.cache_resource
def init_resources():
    # kamus & stopwords
    norm = load_json("https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/nasalsabila_kamus-alay/_json_colloquial-indonesian-lexicon.txt")
    stop = set(load_text("https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/kamus/masdevid_id-stopwords/id.stopwords.02.01.2016.txt"))
    # stemmer
    stemmer = StemmerFactory().create_stemmer()
    # lexicon
    pos1 = load_lexicon("https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv")
    neg1 = load_lexicon("https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv")
    pos2 = load_lexicon("https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/positive.tsv", header=0)
    neg2 = load_lexicon("https://raw.githubusercontent.com/onpilot/sentimen-bahasa/master/leksikon/inset/negative.tsv", header=0)
    positive = pos1.union(pos2)
    negative = neg1.union(neg2)
    return norm, stop, stemmer, positive, negative
norm_dict, stop_words, stemmer, pos_lex, neg_lex = init_resources()

# ===============================================================
# 2️⃣  PREPROCESSING
# ===============================================================
def clean_text(t):
    if not isinstance(t,str): return ""
    t = re.sub(r"http\S+|@\w+|#"," ",t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()

def normalize_token(tokens): return [norm_dict.get(w, w) for w in tokens]
def remove_stop(tokens): return [w for w in tokens if w not in stop_words and len(w)>1]
def stem_tokens(tokens): return [stemmer.stem(w) for w in tokens]

def tokenize(text): return text.split()

# ===============================================================
# 3️⃣  FILTER POLRI
# ===============================================================
def is_relevant_to_polri(text):
    keywords_polri = [
        'polri','polisi','kepolisian','polda','polres','polsek','polresta',
        'polrestabes','poltabes','kapolri','wakapolri','kapolda','wakapolda',
        'kapolres','wakapolres','kapolsek','wakapolsek','kabid humas',
        'kadiv humas','kasatlantas','kasatreskrim','kasatresnarkoba',
        'kasatintelkam','kasatbinmas','brimob','sabhara','lantas','reskrim',
        'intelkam','tipikor','tipidkor','tipidnarkoba','tipidter','densus',
        'spkt','bareksrim','bhabinkamtibmas','polair','polairud','paminal',
        'propam','polantas','bhayangkara','polwan','polsus'
    ]
    keywords_tni = [
        'tni','tentara','militer','prajurit','kostrad','kopassus','babinsa',
        'batalyon','markas','kodam','koramil','danramil','pangdam',
        'menhan','korem','kesatuan','angkatan darat','angkatan laut','angkatan udara'
    ]
    pattern_polri = r'\b(?:'+'|'.join(keywords_polri)+r')\b'
    pattern_tni = r'\b(?:'+'|'.join(keywords_tni)+r')\b'
    has_polri = bool(re.search(pattern_polri,text))
    has_tni = bool(re.search(pattern_tni,text))
    return has_polri or (has_polri and has_tni)

# ===============================================================
# 4️⃣  LABELING (2 KELAS)
# ===============================================================
def label_sentiment(tokens):
    if not isinstance(tokens,list): return "negatif"
    pos = sum(1 for w in tokens if w in pos_lex)
    neg = sum(1 for w in tokens if w in neg_lex)
    return "positif" if pos>=neg else "negatif"

# ===============================================================
# 5️⃣  PIPELINE PREPROCESS + LABEL
# ===============================================================
def preprocess_pipeline(df, text_col):
    df=df.copy()
    df["cleaned"]=df[text_col].astype(str).apply(clean_text)
    df["case_folded"]=df["cleaned"].str.lower()
    df=df[df["case_folded"].apply(is_relevant_to_polri)]
    df["tokens"]=df["case_folded"].apply(tokenize)
    df["normalized"]=df["tokens"].apply(normalize_token)
    df["no_stop"]=df["normalized"].apply(remove_stop)
    df["stemmed"]=df["no_stop"].apply(stem_tokens)
    df["sentiment"]=df["stemmed"].apply(label_sentiment)
    return df

# ===============================================================
# 6️⃣  MACHINE LEARNING & EVALUATION
# ===============================================================
def train_models(df):
    X=df["case_folded"]
    y=df["sentiment"]
    Xtr,Xts,ytr,yts=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)

    tfidf=TfidfVectorizer(max_features=1500)
    Xtr_tf=tfidf.fit_transform(Xtr)
    Xts_tf=tfidf.transform(Xts)

    nb=MultinomialNB(alpha=0.2).fit(Xtr_tf,ytr)
    svm=SVC(kernel='linear',probability=True).fit(Xtr_tf,ytr)

    preds_nb=nb.predict(Xts_tf); preds_svm=svm.predict(Xts_tf)
    prob_nb=nb.predict_proba(Xts_tf)[:,1]
    prob_svm=svm.predict_proba(Xts_tf)[:,1]

    metrics={}
    for name,pred,prob,model in [("Naive Bayes",preds_nb,prob_nb,nb),
                                 ("SVM Linear",preds_svm,prob_svm,svm)]:
        acc=accuracy_score(yts,pred)
        prec=precision_score(yts,pred,pos_label="positif")
        rec=recall_score(yts,pred,pos_label="positif")
        f1=f1_score(yts,pred,pos_label="positif")
        fpr,tpr,_=roc_curve(yts.map({"negatif":0,"positif":1}),prob)
        roc_auc=auc(fpr,tpr)
        cm=confusion_matrix(yts,pred,labels=["positif","negatif"])
        metrics[name]={"acc":acc,"prec":prec,"rec":rec,"f1":f1,
                       "roc":(fpr,tpr,roc_auc),"cm":cm}
    return metrics,tfidf.get_feature_names_out(),Xtr_tf

# ===============================================================
# 7️⃣  VISUALISASI
# ===============================================================
def plot_cm(cm,model):
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",ax=ax,
                xticklabels=["Positif","Negatif"],
                yticklabels=["Positif","Negatif"])
    ax.set_title(f"Confusion Matrix - {model}")
    st.pyplot(fig)

def plot_roc(metrics):
    fig,ax=plt.subplots()
    for name,v in metrics.items():
        fpr,tpr,roc_auc=v["roc"]
        ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.2f})")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); st.pyplot(fig)

def show_wordcloud(df,label):
    text=" ".join(df[df["sentiment"]==label]["case_folded"])
    if not text.strip(): return
    wc=WordCloud(width=800,height=400,background_color='white',
                 colormap='Greens' if label=="positif" else 'Reds').generate(text)
    st.image(wc.to_array(),caption=f"WordCloud - {label}")

# ===============================================================
# 8️⃣  STREAMLIT UI
# ===============================================================
st.title("📊 Analisis Sentimen Polri (Lexicon + ML)")
tab1,tab2=st.tabs(["📂 Upload File","📝 Analisis Teks Tunggal"])

with tab1:
    upl=st.file_uploader("Unggah CSV/Excel",type=["csv","xlsx"])
    if upl:
        df=pd.read_csv(upl) if upl.name.endswith("csv") else pd.read_excel(upl)
        text_col=st.selectbox("Pilih kolom teks:",df.columns)
        if st.button("🔍 Jalankan Analisis"):
            with st.spinner("Memproses data..."):
                df_proc=preprocess_pipeline(df,text_col)
            st.subheader("Hasil Preprocessing & Labeling (contoh):")
            st.dataframe(df_proc[[text_col,"cleaned","case_folded","sentiment"]].head(10))
            st.write("Distribusi Sentimen:"); st.bar_chart(df_proc["sentiment"].value_counts())

            metrics,terms,Xtf=train_models(df_proc)

            st.subheader("📈 Evaluasi Model")
            eval_table=pd.DataFrame({
                "Model":[m for m in metrics],
                "Akurasi":[f"{v['acc']:.3f}" for v in metrics.values()],
                "Presisi":[f"{v['prec']:.3f}" for v in metrics.values()],
                "Recall":[f"{v['rec']:.3f}" for v in metrics.values()],
                "F1":[f"{v['f1']:.3f}" for v in metrics.values()]
            })
            st.dataframe(eval_table)

            st.subheader("📉 Confusion Matrix")
            for m,v in metrics.items(): plot_cm(v["cm"],m)

            st.subheader("📊 ROC Curve"); plot_roc(metrics)

            st.subheader("🪶 Term Frequency (Top 10)")
            tf_counts=np.array(Xtf.sum(axis=0)).flatten()
            freq_df=pd.DataFrame({"term":terms,"freq":tf_counts}).sort_values("freq",ascending=False).head(10)
            st.dataframe(freq_df)
            fig,ax=plt.subplots(); ax.barh(freq_df["term"],freq_df["freq"]); ax.invert_yaxis()
            st.pyplot(fig)

            st.subheader("☁️ WordCloud")
            col1,col2=st.columns(2)
            with col1: show_wordcloud(df_proc,"positif")
            with col2: show_wordcloud(df_proc,"negatif")

            st.download_button("📥 Unduh hasil CSV",
                               df_proc.to_csv(index=False).encode('utf-8'),
                               file_name="hasil_sentimen_polri.csv")

with tab2:
    txt=st.text_area("Masukkan teks untuk analisis:")
    if st.button("🚀 Analisis Teks"):
        if txt.strip():
            clean=clean_text(txt); toks=stem_tokens(remove_stop(normalize_token(tokenize(clean))))
            lbl=label_sentiment(toks)
            st.write("Teks setelah preprocessing:"); st.info(" ".join(toks))
            st.success(f"Hasil Sentimen: **{lbl.upper()}**")
        else: st.warning("Masukkan teks terlebih dahulu.")
