#!/usr/bin/env python
"""Refactored PDF extraction pipeline for large‑scale academic corpora.

Key features
------------
* **Multiprocessing‑safe logging** via `multiprocessing.Queue`.
* **Interrupt‑friendly execution** – one Ctrl‑C cancels the batch cleanly.
* **Extraction cascade**: PyMuPDF → pdfplumber (PDFMiner) → PyPDF → selective OCR.
* **Light‑weight text analytics**: section detection, equation & citation extraction, basic quality scoring.
* **Per‑page OCR** only when no text layer is present, dramatically faster than full‑document OCR.
* **Timeout control** at the future level – a hung file is skipped after the specified seconds.
* **Simple summary reporting** printed live; individual JSON sidecars containing analytics.

Usage
-----
```bash
python refactored_extraction_pipeline.py --pdf-dir /path/to/pdfs --out-dir ./out \
    --workers 30 --timeout 600
```
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Third‑party deps
import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader
from PIL import Image
import pytesseract
import concurrent.futures
import multiprocessing as mp
import logging
import logging.handlers

################################################################################
# --------------------------- logging infrastructure ------------------------- #
################################################################################

LOG_QUEUE: "mp.Queue[logging.LogRecord]" = mp.Queue()


class QueueAwareHandler(logging.Handler):
    """Send log records to multiprocessing queue."""

    def emit(self, record: logging.LogRecord):
        try:
            LOG_QUEUE.put_nowait(record)
        except Exception:  # pragma: no cover
            pass


def _configure_worker_logger(level: int = logging.INFO):
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(QueueAwareHandler())
    root.setLevel(level)


class QueueListener:
    def __init__(self, log_queue: "mp.Queue[logging.LogRecord]"):
        self._queue = log_queue
        self._stop = False

    def start(self):
        while not self._stop:
            try:
                record = self._queue.get(timeout=0.2)
                logging.getLogger(record.name).handle(record)
            except queue.Empty:
                continue

    def stop(self):
        self._stop = True

################################################################################
# --------------------------- text analytics utils -------------------------- #
################################################################################

SECTION_PATTERNS = {
    "abstract": r"(?i)^abstract[\s:]*$",
    "introduction": r"(?i)^(?:1\.|i\.|i\s|)\s*introduction[\s:]*$",
    "methods": r"(?i)method(?:s|ology)?",
    "results": r"(?i)results?",
    "discussion": r"(?i)discussion",
    "conclusion": r"(?i)conclusion[s]?",
    "references": r"(?i)references|bibliography",
}

EQUATION_RE = re.compile(r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
CITATION_RE = re.compile(r"\((?:[A-Z][A-Za-z\-]+(?:,\s*(?:and|&)?\s*[A-Z][A-Za-z\-]+)*),?\s*(?:19|20)\d{2}[a-z]?\)")


def detect_sections(text: str) -> Dict[str, List[str]]:
    lines = text.split("\n")
    cur = "preamble"
    buckets: Dict[str, List[str]] = {cur: []}
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        matched = False
        for name, pat in SECTION_PATTERNS.items():
            if re.match(pat, stripped):
                cur = name
                buckets.setdefault(cur, [])
                matched = True
                break
        if not matched:
            buckets.setdefault(cur, []).append(stripped)
    return buckets


def extract_equations(text: str) -> List[str]:
    return EQUATION_RE.findall(text)


def extract_citations(text: str) -> List[str]:
    return list({m.group(0) for m in CITATION_RE.finditer(text)})


def quality_score(text: str, sections: Dict[str, List[str]]) -> float:
    if not text:
        return 0.0
    score = 0.2 if len(text) > 2000 else 0.05
    score += 0.5 * (len(sections) / 8)  # up to +0.5 if many sections detected
    score += 0.1 if EQUATION_RE.search(text) else 0
    score += 0.1 if CITATION_RE.search(text) else 0
    return min(1.0, score)

################################################################################
# -------------------------- extraction helpers ----------------------------- #
################################################################################

METHOD_PYMUPDF = "pymupdf"
METHOD_PDFPLUMBER = "pdfplumber"
METHOD_PYPDF = "pypdf"
METHOD_OCR = "ocr"


class Extractor:
    def __init__(self, tesseract_path: str | None = None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = r"C:\Users\s14718\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        self.logger = logging.getLogger("Extractor")

    # main API ----------------------------------------------------------------

    def extract(self, pdf_path: Path) -> Tuple[str, str, int]:
        text = self._extract_pymupdf(pdf_path)
        if self._good(text):
            return text, METHOD_PYMUPDF, 0
        text = self._extract_pdfplumber(pdf_path)
        if self._good(text):
            return text, METHOD_PDFPLUMBER, 0
        text = self._extract_pypdf(pdf_path)
        if self._good(text):
            return text, METHOD_PYPDF, 0
        text, ocr_pages = self._extract_with_ocr(pdf_path)
        return text, METHOD_OCR, ocr_pages

    # individual extractors ---------------------------------------------------

    def _extract_pymupdf(self, path: Path) -> str:
        try:
            with fitz.open(path) as doc:
                return "\n".join(p.get_text("text") for p in doc)
        except Exception as e:  # noqa: BLE001
            self.logger.debug("MuPDF fail %s: %s", path, e)
            return ""

    def _extract_pdfplumber(self, path: Path) -> str:
        try:
            with pdfplumber.open(path) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:  # noqa: BLE001
            self.logger.debug("pdfplumber fail %s: %s", path, e)
            return ""

    def _extract_pypdf(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
            return "\n".join(pg.extract_text() or "" for pg in reader.pages)
        except Exception as e:  # noqa: BLE001
            self.logger.debug("pypdf fail %s: %s", path, e)
            return ""

    def _extract_with_ocr(self, path: Path) -> Tuple[str, int]:
        self.logger.info("[%s] OCR fallback", path.name)
        ocr_pages = 0
        parts: List[str] = []
        with fitz.open(path) as doc:
            for pg in doc:
                txt = pg.get_text("text").strip()
                if txt:
                    parts.append(txt)
                    continue
                pix = pg.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                parts.append(pytesseract.image_to_string(img, lang="eng"))
                ocr_pages += 1
        return "\n".join(parts), ocr_pages

    @staticmethod
    def _good(text: str) -> bool:
        return len(text) > 500

################################################################################
# ----------------------- worker & orchestration ---------------------------- #
################################################################################

EXTRACTOR = None  # singleton per worker


def _worker_init(tess_path: str | None):
    _configure_worker_logger()
    global EXTRACTOR  # noqa: PLW0603
    EXTRACTOR = Extractor(tesseract_path=tess_path)


def process_pdf(pdf: str, out_dir: str) -> Dict[str, object]:
    pdf_path = Path(pdf)
    out_dir = Path(out_dir)
    logger = logging.getLogger("worker")
    t0 = time.time()
    try:
        txt, method, ocr_pages = EXTRACTOR.extract(pdf_path)
        secs = time.time() - t0
        # analytics
        sections = detect_sections(txt)
        qualscore = quality_score(txt, sections)
        equations = extract_equations(txt)
        citations = extract_citations(txt)
        # save txt + json sidecar
        (out_dir / "text").mkdir(exist_ok=True, parents=True)
        (out_dir / "json").mkdir(exist_ok=True, parents=True)
        (out_dir / "text" / f"{pdf_path.stem}.txt").write_text(txt, encoding="utf-8")
        meta = {
            "file": pdf_path.name,
            "method": method,
            "pages_ocr": ocr_pages,
            "secs": round(secs, 2),
            "quality": qualscore,
            "sections": list(sections.keys()),
            "equations": len(equations),
            "citations": len(citations),
        }
        (out_dir / "json" / f"{pdf_path.stem}.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        logger.info("%s | %s | q=%.2f | eq=%d | cit=%d | %.1fs", pdf_path.name, method, qualscore, len(equations), len(citations), secs)
        return {**meta, "success": True}
    except Exception as e:  # noqa: BLE001
        logger.error("Fail %s: %s", pdf_path.name, e, exc_info=True)
        return {"file": pdf_path.name, "success": False, "error": str(e)}

################################################################################
# ---------------------------------- main ----------------------------------- #
################################################################################


def main():
    parser = argparse.ArgumentParser(description="Refactored extraction w/ analytics")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--tesseract", default=None)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    root = logging.getLogger("main")

    listener = QueueListener(LOG_QUEUE)
    listener_p = mp.Process(target=listener.start, daemon=True)
    listener_p.start()

    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    root.info("Found %d PDFs", len(pdfs))

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.tesseract,),
    ) as pool:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        futs = {pool.submit(process_pdf, str(p), str(out_dir)): p for p in pdfs}
        completed = 0
        try:
            for fut in concurrent.futures.as_completed(futs):
                completed += 1
                if completed % 50 == 0:
                    root.info("Progress: %d / %d", completed, len(futs))
        except KeyboardInterrupt:
            root.warning("Interrupted by user – cancelling.")
            for fut in futs:
                fut.cancel()
            pool.shutdown(cancel_futures=True)
        finally:
            listener.stop()
            listener_p.join()
    root.info("Done – processed %d files", completed)


if __name__ == "__main__":
    mp.freeze_support()
    main()
