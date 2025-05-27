# -*- coding: utf-8 -*-
"""
pipeline/preprocessing.py

All metadata I/O, text-cleaning, token counts, sharded‐DTM builds,
and final merge live here—with full INFO-level logs.
"""

from pathlib import Path
import csv, re, unicodedata, pickle, logging
from itertools import count
from collections import Counter, defaultdict

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from pipeline import text_cleaning
# ─── globals ────────────────────────────────────────────────────────────────
logger = logging.getLogger("preproc")
csv.field_size_limit(2_000_000)
_READ_OPTS = dict(
    sep="\t", dtype=str, engine="python",
    quoting=csv.QUOTE_NONE, on_bad_lines="warn",
    encoding_errors="replace",
)
_LEMM     = WordNetLemmatizer()
_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
_counter  = count()
ONE_ROW   = {"title","date","published","project","prog"}

# ─── 1) metadata loader ─────────────────────────────────────────────────────
def _collapse(df: pd.DataFrame) -> pd.DataFrame:
    agg = {}
    for c in df.columns.difference(["paper"]):
        if pd.api.types.is_numeric_dtype(df[c]):
            agg[c] = "min"
        else:
            agg[c] = lambda s: "; ".join(sorted(s.dropna().unique()))
    return df.groupby("paper", as_index=False).agg(agg)

def load_meta(meta_dir: str,
              pa_file: str,
              pa_cols: tuple[str,str],
              with_jel: bool = True) -> pd.DataFrame:
    logger.info(f"[load_meta] Reading TSVs from '{meta_dir}'…")
    mdir = Path(meta_dir)
    m: dict[str,pd.DataFrame] = {}
    for f in mdir.glob("*.tsv"):
        try:
            m[f.stem.lower()] = pd.read_csv(f, **_READ_OPTS)
            logger.info(f"  • Loaded {f.name} ({len(m[f.stem.lower()])} rows)")
        except Exception as e:
            logger.warning(f"[load_meta] {f.name}: {e} – skipping")
            m[f.stem.lower()] = pd.DataFrame()

    # collapse one-row tables
    for k in ONE_ROW & m.keys():
        if not m[k].empty and m[k]["paper"].duplicated().any():
            before = len(m[k])
            m[k] = _collapse(m[k])
            logger.info(f"  • Collapsed '{k}.tsv': {before}→{len(m[k])} rows")

    # unify column name
    for df in m.values():
        if "paper_id" not in df.columns and "paper" in df.columns:
            df.rename(columns={"paper":"paper_id"}, inplace=True)

    # paper–author mapping
    paf = Path(meta_dir) / pa_file
    if not paf.exists():
        raise FileNotFoundError(f"{pa_file} not found in {meta_dir}")
    df_pa = pd.read_csv(paf, **_READ_OPTS)
    p_col, a_col = pa_cols
    if p_col not in df_pa.columns or a_col not in df_pa.columns:
        raise KeyError(f"{pa_file} missing columns {p_col}/{a_col}")
    df_pa = df_pa.rename(columns={p_col:"paper_id", a_col:"author_user"})
    m["paperauth"] = df_pa[["paper_id","author_user"]].drop_duplicates()
    logger.info(f"  • paper–author map: {len(m['paperauth'])} rows")

    # assemble single-row papers
    if "papers" not in m:
        pieces = []
        for k in ONE_ROW:
            if k in m and not m[k].empty:
                pieces.append(m[k].set_index("paper_id"))
        if not pieces:
            raise FileNotFoundError("Need at least title.tsv in metadata")
        m["papers"] = pd.concat(pieces, axis=1).reset_index()
        logger.info(f"  • Built 'papers' table with {len(m['papers'])} rows")

    # final long form
    long = m["paperauth"].merge(m["papers"], on="paper_id", how="left")
    if with_jel and "jel" in m and not m["jel"].empty:
        long = long.merge(m["jel"][["paper_id","jel"]],
                          on="paper_id", how="left")
        logger.info(f"  • Merged JEL codes")
    logger.info(f"[load_meta] Final view has {len(long)} rows")
    return long

# ─── 2) clean & save texts ──────────────────────────────────────────────────
def _basic_clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode()
    return re.sub(r"\s+"," ", text).lower().strip()

def clean_and_save_texts(meta: pd.DataFrame,
                         text_dir: str,
                         out_dir: str,
                         n_jobs: int,
                         checkpoint: Path):
    logger.info(f"[clean] Starting cleaning of {len(meta)} docs → '{out_dir}'")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        logger.info("  ► clean_done checkpoint found, skipping")
        return

    def _worker(idx, row):
        pid      = row["paper_id"]
        in_path  = Path(text_dir) / f"{pid}.txt"
        out_path = out / f"{pid}.txt"
    
        if not in_path.exists():
            logger.warning(f"[clean] Missing '{pid}.txt', skipping")
            return
    
        # --- NEW: single-line, full cleaning ----------------------------------
        clean_txt = text_cleaning.preprocess_one(in_path)
        # ----------------------------------------------------------------------
    
        out_path.write_text(clean_txt, encoding="utf-8")
    
        i = next(_counter)
        if i and i % 5_000 == 0:
            logger.info(f"[clean] {i:,} docs processed")

    Parallel(n_jobs=n_jobs)(
        delayed(_worker)(i, row) for i, row in meta.iterrows()
    )

    checkpoint.write_text("done")
    logger.info(f"[clean] Finished cleaning texts")

# ─── 3) token counts ─────────────────────────────────────────────────────────
def compute_token_counts(clean_dir: str,
                         meta: pd.DataFrame,
                         out_path: Path,
                         checkpoint: Path,
                         n_partitions: int):
    logger.info(f"[count] Counting tokens (global + per-author)…")
    if checkpoint.exists():
        logger.info("  ► counts_done checkpoint found, skipping")
        return

    groups = list(meta.groupby("author_user")["paper_id"].apply(list).items())

    def _count_for(author, pids):
        g, a = Counter(), Counter()
        for pid in pids:
            toks = set((Path(clean_dir)/f"{pid}.txt").read_text().split())
            g.update(toks)
            a.update(toks)
        return g, (author, a)

    results = Parallel(n_jobs=n_partitions)(
        delayed(_count_for)(auth, pids) for auth, pids in groups
    )

    global_counts, author_counts = Counter(), defaultdict(Counter)
    for g, (auth, a) in results:
        global_counts.update(g)
        author_counts[auth].update(a)

    with open(out_path, "wb") as f:
        pickle.dump((global_counts, author_counts), f)
    checkpoint.write_text("done")
    logger.info(f"[count] Saved counts to '{out_path}'")

# ─── 4) per-shard DTM build ──────────────────────────────────────────────────
def build_and_save_shard(shard_idx: int,
                         shard_size: int,
                         meta: pd.DataFrame,
                         clean_dir: str,
                         counts_path: Path,
                         cfg: dict,
                         out_dir: str):
    total = len(cfg["shard_sizes"])
    logger.info(f"[shard {shard_idx+1}/{total}] Building (size={shard_size})…")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    done = out/"done"
    if done.exists():
        logger.info(f"  ► shard {shard_idx} already done, skipping")
        return

    global_counts, _ = pickle.load(open(counts_path, "rb"))
    start, end = shard_idx*shard_size, shard_idx*shard_size+shard_size
    df = meta.iloc[start:end]
    logger.info(f"  • Papers {start}–{end} → {len(df)} rows")

    keep = {t for t,c in global_counts.items() if c >= cfg["min_df"]}
    vec = CountVectorizer(
        vocabulary=list(keep),
        lowercase=False,
        max_df=cfg["max_df"],
        ngram_range=tuple(cfg["ngram_range"]),
    )
    docs = [(Path(clean_dir)/f"{pid}.txt").read_text() for pid in df["paper_id"]]
    X = vec.fit_transform(docs)

    df.to_parquet(out/"meta.parquet")
    sp.save_npz(out/"dtm.npz", X)
    with open(out/"vocab.json","w",encoding="utf-8") as f:
        import json
        json.dump(vec.get_feature_names_out().tolist(), f)
    done.write_text("done")
    logger.info(f"[shard {shard_idx+1}/{total}] Saved to '{out_dir}'")

# ─── 5) merge shards ─────────────────────────────────────────────────────────
def merge_shards(shard_dirs: list[Path], final_dir: str):
    logger.info("[merge] Combining shards…")
    final = Path(final_dir); final.mkdir(parents=True, exist_ok=True)

    metas = [pd.read_parquet(sd/"meta.parquet") for sd in shard_dirs]
    pd.concat(metas, ignore_index=True).to_parquet(final/"meta.parquet")
    logger.info(f"  • Merged metadata ({len(metas)} shards)")

    mats = [sp.load_npz(sd/"dtm.npz") for sd in shard_dirs]
    sp.save_npz(final/"dtm.npz", sp.vstack(mats, format="csr"))
    logger.info("  • Merged DTM matrix")

    vocab = None
    import json
    with open(shard_dirs[0]/"vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(final/"vocab.json","w",encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    logger.info("  • Copied vocab.json")

    logger.info("[merge] All shards merged into final directory")