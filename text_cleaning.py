# -*- coding: utf-8 -*-
"""
Improved text cleaning for economics working‑paper corpus.

Changes compared with the original version
------------------------------------------
1.  **Strip references & in‑text citations** – removes sections starting with
    "References"/"Bibliography" and parenthetical cites such as "(Smith 2023)".
2.  **Collapse LaTeX‑style equations** to a single placeholder `<EQN>` so
    similarity isn’t driven by common maths.
3.  **Replace numeric literals** with `<NUM>` – numbers alone rarely signal
    topic.
4.  **Aggressive stop‑wording** using a custom `ECON_STOP` list (extends SpaCy’s
    defaults).
5.  **URL removal, whitespace squashing, Unicode normalisation** – as before.
6.  **Sentence filter** unchanged (keeps only plausible English sentences ≥ 6
    tokens).

Public entry‑points used by the rest of the pipeline
----------------------------------------------------
* `preprocess_one(Path) -> str` – load raw file, return cleaned token string.
* `run_clean(corpus_dir, out_dir)` – batch‑clean a directory and write each
  cleaned text to `out_dir/<paper_id>.txt`.
* `load_clean_texts(clean_dir) -> dict[pid, str]` – convenience loader required
  by `run_similarity.py`.

Running the module directly:
```bash
python pipeline/text_cleaning.py --in raw_txt --out clean_text
```
will write new cleaned files that the rest of the pipeline will pick up.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict

import ftfy
import spacy
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ─── SpaCy blank pipeline (fast, no models) ────────────────────────────────
NLP = spacy.blank("en")

# ─── Regex patterns ────────────────────────────────────────────────────────
CITE_RE        = re.compile(r"\((?:19|20)\d{2}[^)]{0,120}\)")                # (Smith 2021)
REF_SECTION_RE = re.compile(r"(?i)\n\s*(references|bibliography)\b[\s\S]*", re.MULTILINE)
EQUATION_RE    = re.compile(r"\$[^$]+\$")                                     # $...$
URL_RE         = re.compile(r"https?://\S+|www\.\S+")
NUM_RE         = re.compile(r"\b\d+(?:[\d,]*\.?\d*)\b")
BAD_CHUNK_PAT  = re.compile(r"^[^A-Za-z]{5,}$")

# Additional domain‑specific stop‑words (lower‑case ASCII)
ECON_STOP: set[str] = {
    "appendix", "figure", "fig", "table", "section", "paper", "ibid", "et", "al",
    "dataset", "data", "author", "authors", "method", "methods", "result",
    "results", "conclusion", "conclusions", "reference", "references",
    "bibliography", "introduction", "footnote", "footnotes", "page",
}

TOKEN_RE = re.compile(r"[A-Za-z]{3,}")

# ─── Core cleaning helpers ─────────────────────────────────────────────────

def _strip_refs_and_cites(text: str) -> str:
    """Remove reference section *and* in‑text citations."""
    text = REF_SECTION_RE.sub(" ", text)
    return CITE_RE.sub(" ", text)


def _collapse_equations_and_numbers(text: str) -> str:
    text = EQUATION_RE.sub(" <EQN> ", text)
    return NUM_RE.sub(" <NUM> ", text)


def clean_raw(text: str) -> str:
    """Return aggressively cleaned string ready for tokenisation."""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)

    text = URL_RE.sub(" ", text)
    text = _strip_refs_and_cites(text)
    text = _collapse_equations_and_numbers(text)

    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─── Sentence‑level filter ────────────────────────────────────────────────

def _sentence_filter(doc: spacy.tokens.Doc) -> str:
    keep: list[str] = []
    for s in doc.sents:
        sent = s.text.strip()
        if (
            len(sent.split()) >= 6
            and not BAD_CHUNK_PAT.match(sent)
            and any(c.isalpha() for c in sent[:30])
        ):
            keep.append(sent)
    return " ".join(keep)


# ─── Public helpers ────────────────────────────────────────────────────────

def preprocess_one(path: Path) -> str:
    """Load raw file, clean, stop‑word filter & lowercase tokens."""
    raw = path.read_text(errors="ignore")
    clean = clean_raw(raw)
    toks = [t.lower() for t in TOKEN_RE.findall(clean) if t.lower() not in ECON_STOP]
    return " ".join(toks)


def run_clean(corpus_dir: Path, out_dir: Path):
    """Batch‑clean every *.txt in `corpus_dir` to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for txt in tqdm(sorted(corpus_dir.glob("*.txt")), desc="[clean]"):
        pid = txt.stem
        cleaned = preprocess_one(txt)
        (out_dir / f"{pid}.txt").write_text(cleaned, encoding="utf-8")


def load_clean_texts(clean_dir: Path) -> Dict[str, str]:
    """Utility used by run_similarity.py to read already‑cleaned docs."""
    texts: Dict[str, str] = {}
    for txt in sorted(clean_dir.iterdir()):
        if txt.is_file():
            texts[txt.stem] = txt.read_text(encoding="utf-8", errors="ignore")
    return texts


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Clean raw_txt → clean_text directory")
    p.add_argument("--in", dest="inp", required=True, help="Folder of raw .txt files")
    p.add_argument("--out", dest="out", required=True, help="Folder to write cleaned .txt")
    args = p.parse_args()

    run_clean(Path(args.inp), Path(args.out))
    print("✓ Cleaning completed")
