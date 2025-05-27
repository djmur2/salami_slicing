#!/usr/bin/env python3
# create_text_visuals.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

#==============================================================================
# 0. SETUP & PATHS
#==============================================================================
# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

DATA_PATH    = r"C:\Users\s14718\Desktop\salami_modularized\data"
RESULTS_PATH = os.path.join(DATA_PATH, "extended_results")
FIGURES_PATH = os.path.join(DATA_PATH, "extended_figures")

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

#==============================================================================
# 1. LOAD & NORMALIZE ABSTRACTS
#==============================================================================
def load_abstracts():
    f = os.path.join(RESULTS_PATH, "abstracts_for_wordcloud.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f} – run your Stata step first")
    df = pd.read_csv(f)
    # Normalize column name: text <- abstract, if needed
    if "text" not in df.columns and "abstract" in df.columns:
        df = df.rename(columns={"abstract":"text"})
    if "text" not in df.columns:
        raise KeyError("Could not find 'text' or 'abstract' column in your CSV")
    # Ensure numeric ID is called 'paper'
    if "paper_id" in df.columns and "paper" not in df.columns:
        df = df.rename(columns={"paper_id":"paper"})
    return df[["paper","text"]]

#==============================================================================
# 2. BUILD ECONOMICS STOPWORDS FROM CORPUS
#==============================================================================
def build_econ_stopwords(abst_df, top_n=50):
    def tokenize(t):
        t = str(t).lower()
        t = re.sub(r'[^a-z\s]', ' ', t)
        return t.split()
    tokens = []
    for txt in abst_df["text"].dropna():
        tokens.extend(tokenize(txt))
    freq = Counter(tokens)
    eng = set(stopwords.words("english"))
    econ_noise = [w for w,_ in freq.most_common(200) if w not in eng][:top_n]
    econ_sw = eng.union(econ_noise)
    print(f"Added top {top_n} corpus terms to stoplist:", econ_noise[:10], "…")
    return econ_sw

#==============================================================================
# 3. COMPARISON WORD CLOUDS
#==============================================================================
def create_comparison_wordclouds(econ_sw, sim_thresh=0.95):
    sim_df = pd.read_stata(os.path.join(DATA_PATH, "paper_similarity_dataset_blend.dta"))
    top_pairs = (
        sim_df[(sim_df.authors_share>0) & (sim_df.similarity>sim_thresh)]
        .sort_values("similarity", ascending=False)
        .head(10)
    )
    abs_df = load_abstracts()
    fig, axes = plt.subplots(5,4,figsize=(20,25))
    axes = axes.flatten()
    idx = 0

    for _, row in top_pairs.iterrows():
        for side in ["paper_i","paper_j"]:
            if idx>=20: break
            pid = int(row[side])
            text = abs_df.loc[abs_df.paper==pid, "text"]
            snippet = str(text.iloc[0]) if not text.empty else ""
            wc = WordCloud(
                width=400, height=300, background_color="white",
                stopwords=econ_sw, max_words=50, relative_scaling=0.5
            ).generate(snippet or "<no text>")
            axes[idx].imshow(wc, interpolation="bilinear")
            axes[idx].set_title(f"{side[-1].upper()}{pid}\n(sim={row.similarity:.3f})", fontsize=10)
            axes[idx].axis("off")
            idx += 1
        if idx>=20: break

    for j in range(idx, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Word Clouds for Top-10 Pairs (sim > {sim_thresh})", fontsize=16, weight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_PATH, "wordcloud_comparisons.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("→ Saved comparison word clouds to", out)

#==============================================================================
# 4. AGGREGATE WORD CLOUD
#==============================================================================
def create_aggregate_wordcloud(econ_sw):
    abs_df = load_abstracts()
    all_text = " ".join(abs_df.text.dropna().astype(str))
    wc = WordCloud(
        width=1200, height=800, background_color="white",
        stopwords=econ_sw, max_words=100, relative_scaling=0.5
    ).generate(all_text)
    plt.figure(figsize=(12,8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Common Terms in High-Similarity Economics Papers", fontsize=20, weight="bold", pad=20)
    out = os.path.join(FIGURES_PATH, "aggregate_wordcloud.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("→ Saved aggregate word cloud to", out)

#==============================================================================
# 5. TERM FREQUENCY (TF–IDF) BAR CHART
#==============================================================================
def analyze_term_frequencies(econ_sw):
    abs_df = load_abstracts()
    def clean(t): 
        return " ".join(re.sub(r'[^a-z\s]'," ", str(t).lower()).split())
    abs_df["clean"] = abs_df.text.apply(clean)

    vectorizer = TfidfVectorizer(
        max_features=50,
        stop_words=list(econ_sw),
        ngram_range=(1,3),
        min_df=5
    )
    mat = vectorizer.fit_transform(abs_df.clean)
    names = vectorizer.get_feature_names_out()
    scores = mat.sum(axis=0).A1
    top = sorted(zip(names,scores), key=lambda x: -x[1])[:20]

    terms, vals = zip(*top)
    y = np.arange(len(terms))
    plt.figure(figsize=(8,6))
    plt.barh(y, vals)
    plt.yticks(y, terms)
    plt.gca().invert_yaxis()
    plt.xlabel("TF–IDF Score")
    plt.title("Top Terms in High-Similarity Papers", fontsize=14, weight="bold")
    out = os.path.join(FIGURES_PATH, "term_frequency.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print("→ Saved TF–IDF bar chart to", out)

#==============================================================================
# 6. CLUSTERED & MASKED SIMILARITY HEATMAP
#==============================================================================
def create_similarity_heatmap():
    sim_df = pd.read_stata(os.path.join(DATA_PATH, "paper_similarity_dataset_blend.dta"))
    top = (
        sim_df[(sim_df.authors_share>0)&(sim_df.similarity>0.90)]
        .sort_values("similarity", ascending=False)
        .head(20)
    )
    ids = []
    for col in ["paper_i","paper_j"]:
        for v in top[col]:
            if v not in ids and len(ids)<15:
                ids.append(v)
    n = len(ids)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i+1,n):
            cond = (((sim_df.paper_i==ids[i])&(sim_df.paper_j==ids[j])) |
                    ((sim_df.paper_j==ids[i])&(sim_df.paper_i==ids[j])))
            arr = sim_df.loc[cond,"similarity"].values
            if arr.size:
                mat[i,j] = mat[j,i] = arr[0]

    # cluster + reorder
    from scipy.cluster.hierarchy import linkage, leaves_list
    link = linkage(mat, method="average")
    order = leaves_list(link)
    mat_o = mat[np.ix_(order,order)]
    labs  = [f"Paper {ids[k]}" for k in order]
    mask = np.eye(n, dtype=bool)

    plt.figure(figsize=(10,8))
    sns.heatmap(
        mat_o, mask=mask, cmap="viridis", annot=True, fmt=".2f",
        xticklabels=labs, yticklabels=labs,
        cbar_kws={"label":"Similarity"}, square=True
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Clustered Text Similarity (Top 15)", fontsize=16, weight="bold")
    out = os.path.join(FIGURES_PATH, "similarity_heatmap_clustered.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print("→ Saved clustered similarity heatmap to", out)

#==============================================================================
# 7. COAUTHOR NETWORK ANALYSIS
#==============================================================================
def analyze_coauthor_network():
    csvf = os.path.join(FIGURES_PATH, "coauthor_network.csv")
    if not os.path.exists(csvf):
        print(f"⛔ Missing {csvf} — run your Stata author‐network export first.")
        return

    df = pd.read_csv(csvf)
    # degree = total collaborations per author
    deg_i = df.groupby(['author_i','name_i'])['collab_count'].sum().reset_index()
    deg_i.columns = ['author_id','author_name','collab_count']
    deg_j = df.groupby(['author_j','name_j'])['collab_count'].sum().reset_index()
    deg_j.columns = ['author_id','author_name','collab_count']
    deg = pd.concat([deg_i,deg_j]).groupby(['author_id','author_name'])['collab_count'].sum().reset_index()

    top10 = deg.sort_values('collab_count', ascending=False).head(10)
    print("\nTop 10 Authors by Total Collaborations:")
    print(top10.to_string(index=False))

    # bar chart
    plt.figure(figsize=(8,6))
    plt.barh(top10['author_name'][::-1], top10['collab_count'][::-1])
    plt.xlabel("Total Collaborations")
    plt.title("Top 10 Authors by Coauthorship Degree")
    plt.tight_layout()
    out = os.path.join(FIGURES_PATH, "coauthor_degree_top10.png")
    plt.savefig(out, dpi=300)
    print("→ Saved coauthor‐degree bar chart to", out)

#==============================================================================
# MAIN
#==============================================================================
if __name__ == "__main__":
    print("1. Building abstracts corpus…")
    abstracts = load_abstracts()
    econ_sw   = build_econ_stopwords(abstracts)

    print("\n2. Word‐cloud comparisons…")
    create_comparison_wordclouds(econ_sw)

    print("\n3. Aggregate word‐cloud…")
    create_aggregate_wordcloud(econ_sw)

    print("\n4. Term‐frequency analysis…")
    analyze_term_frequencies(econ_sw)

    print("\n5. Similarity heatmap…")
    create_similarity_heatmap()

    print("\n6. Coauthor network analysis…")
    analyze_coauthor_network()

    print("\nAll visualizations complete. Check:", FIGURES_PATH)