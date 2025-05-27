# embeddings.py
"""
pipeline/embeddings.py

TF-IDF → cosine similarity in parallel, batched.
Fully parallelized for efficient processing of large document collections.
"""
import logging
import time
import numpy as np
import psutil
import scipy.sparse as sp
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed, parallel_backend
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _cos_batch(start: int, end: int, X: sp.csr_matrix):
    """Compute cosine similarity for a batch of documents against all documents."""
    sims = cosine_similarity(X[start:end], X).astype(np.float32)
    return start, sims

def cosine_sim_batched(
    X: sp.csr_matrix,
    batch: int = 2000,
    n_jobs: int = None
) -> np.ndarray:
    """
    Compute cosine similarity matrix in batches, using parallel processing.
    
    Args:
        X: TF-IDF or other feature matrix (n_documents × n_features)
        batch: Batch size for processing
        n_jobs: Number of parallel jobs
        
    Returns:
        Full similarity matrix (n_documents × n_documents)
    """
    n = X.shape[0]
    sim_full = np.zeros((n, n), dtype=np.float32)
    batches = [(i, min(i + batch, n)) for i in range(0, n, batch)]
    logger.info(f"Cosine similarity in {len(batches)} batches...")

    # Determine optimal n_jobs if not specified
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) - 1)
    
    logger.info(f"Using {n_jobs} parallel jobs")
    
    with parallel_backend("loky", n_jobs=n_jobs):
        for start, sims in Parallel()(
            delayed(_cos_batch)(s, e, X) for s, e in tqdm(batches, desc="[SIM]", unit="batch")
        ):
            sim_full[start:start + sims.shape[0], :] = sims
            
    # Ensure the matrix is symmetric and has ones on the diagonal
    np.fill_diagonal(sim_full, 1.0)
    return sim_full

def build_text_embeddings(
    clean_texts: Dict[str, str],
    max_features: int = 50000,
    chunk_size: int = 2000,
    n_jobs: int = None
) -> Tuple[List[str], np.ndarray]:
    """
    Build TF-IDF embeddings and compute cosine similarity in parallel.
    
    Args:
        clean_texts: Dictionary mapping paper_id to cleaned text
        max_features: Maximum number of TF-IDF features
        chunk_size: Batch size for similarity computation
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (paper_ids, similarity_matrix)
    """
    paper_ids = list(clean_texts.keys())
    corpus = [clean_texts[pid] for pid in paper_ids]

    logger.info(f"TF-IDF vectorizing {len(corpus):,} docs...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        max_df=0.9,
        min_df=5,
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english"
    )
    X = tfidf.fit_transform(corpus)
    logger.info(f"TF-IDF matrix {X.shape}, nnz={X.nnz:,} in {time.time()-t0:.1f}s")

    logger.info(f"Computing cosine similarity with chunk_size={chunk_size}, n_jobs={n_jobs}...")
    t0 = time.time()
    sim = cosine_sim_batched(X, batch=chunk_size, n_jobs=n_jobs)
    logger.info(f"Similarity matrix computed in {time.time()-t0:.1f}s")
    
    return paper_ids, sim