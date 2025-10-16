# cna_rag_agent/utils/oa.py
import os, re, time, pathlib, requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote

UNPAYWALL_API = "https://api.unpaywall.org/v2/"
USER_AGENT = "CNfA-RAG-Agent/1.0 (+https://example.org)"

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return re.sub(r'-+', '-', s).strip('-')[:120] or "paper"

def fetch_oa_pdfs(
    dois: List[str],
    email: Optional[str] = None,
    save_dir: str = "downloads",
    throttle_sec: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Resolve each DOI via Unpaywall and download the best OA PDF when available.
    Returns: list of rows {doi, title, source, saved_path, download_url, status, note}
    """
    email = email or os.getenv("UNPAYWALL_EMAIL") or os.getenv("STUDENT_EMAIL")
    if not email:
        return [{"doi": d, "status": "error", "note": "No email configured (UNPAYWALL_EMAIL)."} for d in dois]

    out_rows = []
    save_root = pathlib.Path(save_dir); save_root.mkdir(parents=True, exist_ok=True)

    for d in dois:
        row = {"doi": d, "title": "", "source": "", "saved_path": "", "download_url": "", "status": "", "note": ""}
        try:
            meta_url = f"{UNPAYWALL_API}{quote(d)}?email={quote(email)}"
            r = requests.get(meta_url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code != 200:
                row.update(status="error", note=f"unpaywall {r.status_code}")
            else:
                meta = r.json()
                row["title"] = meta.get("title") or ""
                row["source"] = (meta.get("journal_name") or meta.get("publisher") or "").strip()
                best = meta.get("best_oa_location") or {}
                pdf_url = best.get("url_for_pdf") or best.get("url") or ""
                row["download_url"] = pdf_url or ""
                if not pdf_url:
                    row.update(status="no_oa", note="No OA PDF link from Unpaywall")
                else:
                    fname = f"{_slugify(row['title'] or d)}.pdf"
                    fpath = save_root / fname
                    with requests.get(pdf_url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60) as pr:
                        if pr.status_code == 200 and "pdf" in pr.headers.get("Content-Type","").lower():
                            with open(fpath, "wb") as fh:
                                for chunk in pr.iter_content(chunk_size=8192):
                                    if chunk: fh.write(chunk)
                            row["saved_path"] = str(fpath)
                            row["status"] = "downloaded"
                        else:
                            row.update(status="error", note=f"pdf fetch {pr.status_code} / {pr.headers.get('Content-Type')}")
            out_rows.append(row)
        except Exception as e:
            row.update(status="error", note=str(e))
            out_rows.append(row)
        time.sleep(throttle_sec)
    return out_rows
