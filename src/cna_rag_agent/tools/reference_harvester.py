# -*- coding: utf-8 -*-
"""
Reference harvester: extract DOIs from a report JSON, resolve/match DOIs,
retrieve legal OA PDFs via Unpaywall, fall back to DOI resolve + landing-page PDF discovery,
ZIP results, and emit a manifest. Paywalled items are skipped.

Requires:
  pip install requests beautifulsoup4
"""

from __future__ import annotations
import os
import re
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

from cna_rag_agent.utils.logging_config import logger

try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:  # pragma: no cover
    BeautifulSoup = None

# ---- Config ----
USER_AGENT = "CNfA-Research-CoPilot/1.1 (+https://example.com) requests"
TIMEOUT = 20
PDF_MAX_BYTES = 50 * 1024 * 1024  # 50 MB safety
CROSSREF_API = "https://api.crossref.org/works"
UNPAYWALL_API = "https://api.unpaywall.org/v2"

SAFE_CHARS = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

DOI_REGEX = re.compile(
    r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
)

def _safe_basename(name: str, suffix: str = "") -> str:
    filtered = "".join(c for c in name if c in SAFE_CHARS).strip().replace(" ", "_")
    if not filtered:
        filtered = f"file_{int(time.time())}"
    if suffix and not filtered.lower().endswith(suffix.lower()):
        filtered += suffix
    return filtered

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _http(method: str, url: str, **kwargs) -> Optional[requests.Response]:
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", USER_AGENT)
    try:
        r = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
        return r
    except Exception as e:
        logger.warning(f"[harvest] {method} failed for {url}: {e}")
        return None

def _looks_like_pdf_url(url: str) -> bool:
    u = (url or "").lower()
    return u.endswith(".pdf")

def _resolve_doi_to_url(doi: str) -> Optional[str]:
    doi = doi.strip()
    if not doi:
        return None
    if doi.lower().startswith("http"):
        return doi
    url = f"https://doi.org/{doi}"
    r = _http("GET", url, allow_redirects=True)
    if not r:
        return None
    if r.history and r.url:
        return r.url
    if r.status_code < 400:
        return r.url
    return None

def _find_pdf_on_page(page_url: str) -> Optional[str]:
    if not BeautifulSoup:
        return None
    r = _http("GET", page_url)
    if not r or r.status_code >= 400:
        return None
    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return None

    # <link rel="alternate" type="application/pdf"> or *.pdf
    for link in soup.find_all("link"):
        typ = (link.get("type") or "").lower()
        href = link.get("href")
        if not href:
            continue
        if "application/pdf" in typ or href.lower().endswith(".pdf"):
            return requests.compat.urljoin(page_url, href)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            return requests.compat.urljoin(page_url, href)

    return None

def _download_pdf(url: str, out_dir: Path, hint_name: str = "") -> Optional[Path]:
    _ensure_dir(out_dir)
    name_base = _safe_basename(hint_name or Path(url).name, suffix=".pdf")
    out_path = out_dir / name_base

    with requests.get(url, stream=True, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as r:
        if r.status_code >= 400:
            return None
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > PDF_MAX_BYTES:
                    logger.warning(f"[harvest] skipping large file (>50MB) {url}")
                    return None
                f.write(chunk)
    return out_path

def _candidate_name_from_citation(full_citation: str) -> str:
    try:
        year = re.search(r"\((\d{4})\)", full_citation)
        year = year.group(1) if year else ""
        first = full_citation.split(".")[0]  # authors
        title_part = full_citation.split(".")[1].strip() if "." in full_citation else ""
        return f"{first}_{year}_{title_part}"[:120]
    except Exception:
        return "paper"

# ---------- DOI extraction / enrichment ----------

def extract_dois_from_report(report: Dict) -> List[Dict[str, Optional[str]]]:
    """
    From a report dict (your JSON), return a list of dicts:
      { "full_citation": str, "doi": str|None, "url": str|None }
    We also regex-scan the full_citation for a DOI if the 'doi' field is empty.
    """
    out = []
    refs = (report or {}).get("references", []) or []
    for r in refs:
        full = r.get("full_citation") or ""
        doi = (r.get("doi") or "").strip() or None
        url = (r.get("url") or "").strip() or None

        if not doi:
            m = DOI_REGEX.search(full)
            if m:
                doi = m.group(1)

        out.append({"full_citation": full, "doi": doi, "url": url})
    return out

def crossref_find_doi_from_citation(full_citation: str) -> Optional[str]:
    """
    Last-resort attempt: ask Crossref for a bibliographic match.
    """
    q = full_citation.strip()
    if not q:
        return None
    params = {"query.bibliographic": q, "rows": 1}
    r = _http("GET", CROSSREF_API, params=params)
    if not r or r.status_code >= 400:
        return None
    try:
        data = r.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return None
        doi = items[0].get("DOI")
        return doi
    except Exception:
        return None

# ---------- Unpaywall ----------

def unpaywall_pdf_url(doi: str, email: str) -> Optional[str]:
    """
    Return the best OA PDF URL (if any) for a DOI via Unpaywall.
    """
    if not doi or not email:
        return None
    doi = doi.strip()
    url = f"{UNPAYWALL_API}/{doi}"
    r = _http("GET", url, params={"email": email})
    if not r or r.status_code >= 400:
        return None
    try:
        data = r.json()
        # Prefer a direct PDF if available
        pol = data.get("best_oa_location") or {}
        pdf = pol.get("url_for_pdf") or pol.get("url")
        if pdf and _looks_like_pdf_url(pdf):
            return pdf
        # Search other OA locations too
        for loc in data.get("oa_locations", []) or []:
            cand = loc.get("url_for_pdf") or loc.get("url")
            if cand and _looks_like_pdf_url(cand):
                return cand
        return None
    except Exception:
        return None

# ---------- Public API ----------

def harvest_reference_pdfs_oa(
    report: Dict,
    work_dir: Path,
    topic_safe: str,
    unpaywall_email: Optional[str] = None,
    try_crossref: bool = True,
) -> Tuple[List[Path], Dict]:
    """
    OA-first workflow:
    1) Extract DOIs from report (and regex).
    2) If missing DOI and try_crossref: attempt Crossref match.
    3) If DOI present and unpaywall_email given: try Unpaywall for direct PDF.
    4) Else: DOI resolve + landing page scrape.
    5) Else, if a URL exists: check for .pdf or scrape page for PDF.
    Returns (downloaded_paths, manifest).
    """
    topic_dir = work_dir / "downloads" / topic_safe
    _ensure_dir(topic_dir)

    refs = extract_dois_from_report(report)

    successes: List[Path] = []
    manifest = {"topic": topic_safe, "items": []}

    for r in refs:
        full = r.get("full_citation") or ""
        doi = (r.get("doi") or "").strip() or None
        url = (r.get("url") or "").strip() or None
        best_pdf: Optional[str] = None
        method = None

        # Try to get a DOI from Crossref if missing
        if not doi and try_crossref:
            doi = crossref_find_doi_from_citation(full)
            if doi:
                method = "crossref_doi"

        # Unpaywall OA PDF
        if doi and unpaywall_email:
            pdf = unpaywall_pdf_url(doi, unpaywall_email)
            if pdf:
                best_pdf = pdf
                method = method or "unpaywall_pdf"

        # DOI resolve -> scrape
        if not best_pdf and doi:
            landing = _resolve_doi_to_url(doi)
            if landing:
                if _looks_like_pdf_url(landing):
                    best_pdf = landing
                    method = method or "doi_resolve_pdf"
                else:
                    pdf2 = _find_pdf_on_page(landing)
                    if pdf2:
                        best_pdf = pdf2
                        method = method or "doi_resolve_scrape"

        # URL fallback
        if not best_pdf and url:
            if _looks_like_pdf_url(url):
                best_pdf = url
                method = method or "url_pdf"
            else:
                pdf3 = _find_pdf_on_page(url)
                if pdf3:
                    best_pdf = pdf3
                    method = method or "url_scrape"

        record = {"full_citation": full, "doi": doi, "url": url, "pdf_url": best_pdf, "status": "skipped", "method": method}

        if best_pdf:
            hint = _candidate_name_from_citation(full) or "paper"
            local = _download_pdf(best_pdf, topic_dir, hint_name=hint)
            if local:
                record["status"] = "downloaded"
                record["local_path"] = str(local)
                successes.append(local)

        manifest["items"].append(record)

    with open(topic_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return successes, manifest


def zip_reference_folder(work_dir: Path, topic_safe: str) -> Path:
    folder = work_dir / "downloads" / topic_safe
    zip_path = work_dir / "downloads" / f"{topic_safe}.zip"
    _ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(folder))
    return zip_path
