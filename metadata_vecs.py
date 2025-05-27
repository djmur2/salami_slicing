# -*- coding: utf-8 -*-
"""
pipeline/metadata_vecs.py

Builds metadata-based similarity from:
  • JEL codes (one-hot)
  • published_text flag
(no refs, no full journal one-hot)
"""
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def _prep_meta_lookup(
    paper_ids: list[str],
    meta_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    # keep only our papers + needed cols
    m = (
        meta_df
        .loc[meta_df["paper_id"].isin(paper_ids), ["paper_id", "jel", "published_text"]]
        .set_index("paper_id", drop=False)
    )

    # 1) JEL one-hot
    def _split_jels(x):
        if isinstance(x, list):
            return [str(el) for el in x if el]
        if pd.isna(x) or not str(x).strip():
            return []
        return [code for code in re.split(r"\s*;\s*", str(x)) if code]

    jel_series = m["jel"].apply(_split_jels)
    all_jels   = sorted({code for codes in jel_series for code in codes})
    jel_idx    = {code: i for i, code in enumerate(all_jels)}

    J = np.zeros((len(paper_ids), len(all_jels)), dtype=int)
    for row, pid in enumerate(paper_ids):
        for code in jel_series.get(pid, []):
            idx = jel_idx.get(code)
            if idx is not None:
                J[row, idx] = 1

    # 2) published_text flag
    P = np.zeros((len(paper_ids), 1), dtype=int)
    for row, pid in enumerate(paper_ids):
        if pid in m.index:
            text = m.at[pid, "published_text"]
            if isinstance(text, str) and text.strip():
                P[row, 0] = 1

    return J, P


def build_metadata_similarity(
    paper_ids: list[str],
    meta_df: pd.DataFrame,
    jel_weight: float = 0.8,
    pub_weight: float = 0.2
) -> np.ndarray:
    """
    Returns an (n_papers × n_papers) cosine-similarity matrix
    built from [JEL one-hot | published_flag].
    
    Args:
        paper_ids: List of paper IDs to include in the matrix
        meta_df: Metadata DataFrame with paper_id, jel, and published_text columns
        jel_weight: Weight for JEL similarity (default: 0.8)
        pub_weight: Weight for publication status similarity (default: 0.2)
        
    Returns:
        np.ndarray: Combined metadata similarity matrix
    """
    logger.info(f"Building metadata similarity for {len(paper_ids)} papers")
    
    # Get JEL and publication vectors
    J, P = _prep_meta_lookup(paper_ids, meta_df)
    
    # If we have no JEL codes, handle gracefully
    if J.shape[1] == 0:
        logger.warning("No JEL codes found - metadata similarity will be based only on publication status")
        jel_sim = np.zeros((len(paper_ids), len(paper_ids)))
        jel_weight = 0
        pub_weight = 1.0
    else:
        # Compute JEL similarity
        jel_sim = cosine_similarity(J)
        logger.info(f"JEL similarity matrix: {jel_sim.shape}")
    
    # Compute publication similarity (1 if matching status, 0 otherwise)
    if P.sum() == 0:
        logger.warning("No publication status found - metadata similarity will be based only on JEL codes")
        pub_sim = np.zeros((len(paper_ids), len(paper_ids)))
        pub_weight = 0
        jel_weight = 1.0 if J.shape[1] > 0 else 0
    else:
        pub_sim = 1 - np.abs(P - P.T)
        logger.info(f"Publication status similarity matrix: {pub_sim.shape}")
    
    # No metadata at all?
    if jel_weight == 0 and pub_weight == 0:
        logger.warning("No metadata available for similarity calculation - returning identity matrix")
        return np.eye(len(paper_ids))
    
    # Normalize weights if needed
    total = jel_weight + pub_weight
    if total > 0:
        jel_weight /= total
        pub_weight /= total
    
    # Combine similarity matrices
    logger.info(f"Combining similarity matrices (JEL: {jel_weight:.2f}, Pub: {pub_weight:.2f})")
    combined = (jel_weight * jel_sim) + (pub_weight * pub_sim)
    
    return combined