# -*- coding: utf-8 -*-
import streamlit as st
import sys
from pathlib import Path
from uuid import uuid4
import re
import requests
from urllib.parse import quote
from io import BytesIO
from typing import List, Dict, Any, Optional
import os
from cna_rag_agent.utils.oa import fetch_oa_pdfs

# --- Setup Project Path ---
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- End Setup Project Path ---

st.set_page_config(
    page_title="CNfA Research Co-Pilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Imports (stable base) ---
try:
    from cna_rag_agent.utils.logging_config import logger
    from cna_rag_agent.vector_store.store_manager import get_persistent_client
    from cna_rag_agent.config import CHROMA_COLLECTION_NAME
except Exception as e:
    st.error(f"Base imports failed: {e}")
    raise

# Router (optional – we keep it if present)
router_available = True
try:
    from cna_rag_agent.agent.orchestrator import create_advanced_agent_router
except Exception as e:
    router_available = False
    logger.warning(f"Orchestrator not available; chat fallback only. Reason: {e}")

# Graph (optional)
graph_available = True
try:
    from cna_rag_agent.graph_store.query import create_graph_query_chain
except Exception as e:
    graph_available = False
    logger.warning(f"Graph chain import failed (optional): {e}")

# Specialists (use your *current* API from specialists.py)
try:
    from cna_rag_agent.agent.specialists import (
        # heavy pipeline pieces
        harvest_bibliography,
        compose_report,
        expand_if_thin,
        save_report_to_docx,
        # tool-using agent
        create_professional_researcher_agent,
    )
except Exception as e:
    st.error(f"Failed to import specialists: {e}")
    st.stop()

# Embedding + ingestion helpers
# We'll embed and add to your existing Chroma collection directly.
try:
    from cna_rag_agent.embedder import get_embedding_model
except Exception as e:
    get_embedding_model = None
    logger.warning(f"Embedder import failed (ingest still tries best): {e}")

try:
    import PyPDF2
except Exception as e:
    PyPDF2 = None
    logger.warning(f"PyPDF2 not available; PDF text extraction limited: {e}")


# ---------------- Session + Cache ----------------

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid4())

@st.cache_resource
def get_agent():
    if not router_available:
        return None
    logger.info("Streamlit: Initializing advanced agent router...")
    agent_executor = create_advanced_agent_router()
    logger.info("Streamlit: Agent router initialized and cached.")
    return agent_executor

@st.cache_resource
def get_graph_chain_or_none():
    if not graph_available:
        return None
    logger.info("Streamlit: Initializing graph query chain (optional)...")
    try:
        chain = create_graph_query_chain()
        return chain
    except Exception as e:
        logger.warning(f"Graph not available: {e}")
        return None


# ---------------- Utilities ----------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

REPORTS_DIR = ensure_dir(Path("generated_reports"))
PAPERS_DIR = ensure_dir(Path("downloaded_papers"))

def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in (" ", "_", "-", ".") else "_" for ch in name).strip().replace(" ", "_")

def is_slash_command(txt: str) -> bool:
    return txt.strip().startswith("/")

def parse_command(txt: str):
    parts = txt.strip().split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    return cmd, arg

def as_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ---------------- Report Generation using your specialists.py ----------------

def generate_report_with_specialist(topic: str) -> Dict[str, Any]:
    """
    Matches your specialists.py pipeline:
      - create the heavy tool-using agent
      - harvest bibliography
      - compose report
      - expand if thin
    Returns the JSON dict for the report.
    """
    agent = create_professional_researcher_agent()
    refs = harvest_bibliography(agent, topic)
    report = compose_report(agent, topic, refs)
    report = expand_if_thin(agent, report)
    return report


# ---------------- DOI / URL Extraction ----------------

DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[^\s\]\);,>]+", re.IGNORECASE)

def normalize_doi(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return s.strip().strip(".,);]»››>")

def extract_dois_urls_from_report(report: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    """
    From the report JSON, pull candidate DOIs and URLs from:
      - references array
      - title/introduction/conclusion text (just in case)
    Returns list of dicts: { 'full_citation', 'doi', 'url' }
    """
    out = []
    seen = set()

    def add_item(cite: str, doi: Optional[str], url: Optional[str]):
        key = (normalize_doi(doi) or (url or "") or (cite or "")).lower()
        if key and key not in seen:
            seen.add(key)
            out.append({"full_citation": cite or "", "doi": normalize_doi(doi) if doi else None, "url": url})

    # From references
    for r in (report.get("references") or []):
        cite = (r.get("full_citation") or "").strip()
        doi = (r.get("doi") or "").strip() or None
        url = (r.get("url") or "").strip() or None

        if not doi and cite:
            m = DOI_REGEX.search(cite)
            if m:
                doi = normalize_doi(m.group(0))

        if not doi and url and "doi.org/" in url:
            try:
                doi = normalize_doi(url.split("doi.org/")[1])
            except Exception:
                pass

        add_item(cite, doi, url)

    # Scan body text
    body_fields = [
        report.get("title", ""),
        report.get("introduction", ""),
        report.get("key_findings_and_methodologies", ""),
        report.get("conclusion_and_future_directions", "")
    ]
    for sec in (report.get("theoretical_foundations") or []):
        body_fields.append(sec.get("content", ""))

    for text in body_fields:
        if not text:
            continue
        for m in DOI_REGEX.finditer(text):
            doi = normalize_doi(m.group(0))
            add_item("", doi, None)

    return out

def list_dois_from_report(report: Dict[str, Any]) -> List[str]:
    items = extract_dois_urls_from_report(report)
    dois = []
    seen = set()
    for it in items:
        d = normalize_doi(it.get("doi") or "")
        if d and d not in seen:
            seen.add(d)
            dois.append(d)
    return dois


# ---------------- Unpaywall + Crossref + Fetch PDFs (legal OA only) ----------------

def unpaywall_best_pdf_url(doi: str, student_email: str) -> Optional[str]:
    """
    Query Unpaywall for an OA PDF URL (if available).
    """
    if not student_email or "@" not in student_email:
        return None
    try:
        url = f"https://api.unpaywall.org/v2/{quote(doi)}?email={quote(student_email)}"
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url and data.get("oa_locations"):
            for loc in data["oa_locations"]:
                if loc.get("url_for_pdf"):
                    pdf_url = loc["url_for_pdf"]
                    break
                if not pdf_url and str(loc.get("url", "")).lower().endswith(".pdf"):
                    pdf_url = loc["url"]
        return pdf_url
    except Exception as e:
        logger.warning(f"Unpaywall error for DOI {doi}: {e}")
        return None

def crossref_meta(doi: str) -> Dict[str, Optional[str]]:
    """
    Fetch title & source (journal) via Crossref for display.
    """
    try:
        r = requests.get(f"https://api.crossref.org/works/{quote(doi)}", timeout=15)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            title = " ".join(msg.get("title", [])).strip() or None
            source = " ".join(msg.get("container-title", [])).strip() or None
            return {"title": title, "source": source}
    except Exception as e:
        logger.warning(f"Crossref error for {doi}: {e}")
    return {"title": None, "source": None}

def try_direct_pdf(url: str) -> Optional[bytes]:
    """
    If a URL looks directly downloadable as PDF, try to fetch it.
    """
    if not url:
        return None
    try:
        headers = {"User-Agent": "CNfA-Student/1.0"}
        r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "").lower()
        if r.status_code == 200 and ("application/pdf" in ctype or url.lower().endswith(".pdf")):
            return r.content
        if r.status_code == 200 and "application/octet-stream" in ctype:
            return r.content
        return None
    except Exception as e:
        logger.warning(f"Direct PDF fetch failed for {url}: {e}")
        return None

def fetch_pdfs_for_report(report: Dict[str, Any], student_email: str) -> Dict[str, Any]:
    """
    Resolve & download PDFs for all DOIs/URLs found in a report.
    Saves into PAPERS_DIR. Returns summary with successes/failures.
    """
    items = extract_dois_urls_from_report(report)
    successes, failures = [], []

    for i, it in enumerate(items, 1):
        doi = normalize_doi(it.get("doi") or "")
        url = it.get("url")
        label = doi or url or f"ref_{i}"

        pdf_bytes = None
        source_used = None
        pdf_url_used = None

        if doi:
            pdf_url = unpaywall_best_pdf_url(doi, student_email)
            if pdf_url:
                pdf_bytes = try_direct_pdf(pdf_url)
                if pdf_bytes:
                    source_used = "Unpaywall"
                    pdf_url_used = pdf_url

        if not pdf_bytes and url:
            if "doi.org/" not in url:
                pb = try_direct_pdf(url)
                if pb:
                    pdf_bytes = pb
                    source_used = "Direct"
                    pdf_url_used = url

        if pdf_bytes:
            fname = safe_filename(f"{(doi or 'no-doi')}.pdf") if doi else safe_filename(f"url_{i}.pdf")
            out_path = PAPERS_DIR / fname
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            meta = crossref_meta(doi) if doi else {"title": None, "source": None}
            successes.append({
                "doi": doi or None,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "label": label,
                "path": str(out_path),
                "pdf_url": pdf_url_used,
                "method": source_used
            })
        else:
            meta = crossref_meta(doi) if doi else {"title": None, "source": None}
            failures.append({
                "doi": doi or None,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "label": label,
                "reason": "No OA PDF found via Unpaywall/direct"
            })
    return {"downloaded": successes, "failed": failures}

def fetch_pdfs_for_dois(dois: List[str], student_email: str) -> Dict[str, Any]:
    """
    Same as fetch_pdfs_for_report, but for explicit DOI list.
    """
    successes, failures = [], []
    for doi_in in dois:
        doi = normalize_doi(doi_in)
        pdf_bytes = None
        pdf_url_used = None
        method = None

        if not doi:
            failures.append({"doi": None, "label": doi_in, "title": None, "source": None, "reason": "Invalid DOI"})
            continue

        pdf_url = unpaywall_best_pdf_url(doi, student_email)
        if pdf_url:
            pdf_bytes = try_direct_pdf(pdf_url)
            if pdf_bytes:
                pdf_url_used = pdf_url
                method = "Unpaywall"

        if pdf_bytes:
            fname = safe_filename(f"{doi}.pdf")
            out_path = PAPERS_DIR / fname
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)
            meta = crossref_meta(doi)
            successes.append({
                "doi": doi,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "label": doi,
                "path": str(out_path),
                "pdf_url": pdf_url_used,
                "method": method
            })
        else:
            meta = crossref_meta(doi)
            failures.append({
                "doi": doi,
                "title": meta.get("title"),
                "source": meta.get("source"),
                "label": doi,
                "reason": "No OA PDF found via Unpaywall/direct"
            })
    return {"downloaded": successes, "failed": failures}


# ---------------- Ingest PDFs into Chroma ----------------

def chunk_text(txt: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    n = max(50, chunk_size)
    ov = max(0, min(overlap, n - 10))
    while i < len(txt):
        chunk = txt[i:i+n]
        chunks.append(chunk)
        i += n - ov
    return chunks

def extract_text_from_pdf(path: Path) -> str:
    if not PyPDF2:
        return ""
    try:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text() or ""
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        logger.warning(f"PDF extract failed for {path}: {e}")
        return ""

def ingest_pdfs_to_vector_store(pdf_paths: List[str]) -> Dict[str, Any]:
    """
    Adds chunks to your existing persistent Chroma collection with Gemini embeddings.
    """
    try:
        client = get_persistent_client()
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        return {"ok": False, "reason": f"Chroma not available: {e}"}

    if get_embedding_model is None:
        return {"ok": False, "reason": "Embedding model not available."}
    try:
        embedder = get_embedding_model()
    except Exception as e:
        return {"ok": False, "reason": f"Embedding init failed: {e}"}

    added = 0
    errors = []
    for p in pdf_paths:
        pth = Path(p)
        if not pth.exists():
            errors.append({"path": p, "reason": "File not found"})
            continue
        text = extract_text_from_pdf(pth)
        if not text.strip():
            errors.append({"path": p, "reason": "No text extracted"})
            continue

        chunks = chunk_text(text)
        if not chunks:
            errors.append({"path": p, "reason": "No chunks created"})
            continue

        try:
            vectors = embedder.embed_documents(chunks)
            ids = [f"{pth.name}:{i}" for i in range(len(chunks))]
            metas = [{"source": str(pth), "chunk": i} for i in range(len(chunks))]
            collection.add(ids=ids, embeddings=vectors, metadatas=metas, documents=chunks)
            added += len(chunks)
        except Exception as e:
            errors.append({"path": p, "reason": f"Chroma add failed: {e}"})

    return {"ok": True, "added_chunks": added, "errors": errors}


# ---------------- Status Table Rendering ----------------

def render_pdf_status_table(results: Dict[str, Any], show_buttons: bool = True):
    """
    Render a status table for downloaded/failed PDFs with Download buttons.
    """
    rows_md = []
    header = "| DOI | Title | Source | Saved Path | Download Link |\n|---|---|---|---|---|"
    rows_md.append(header)

    # Downloaded
    for item in results.get("downloaded", []):
        doi = item.get("doi") or "—"
        title = (item.get("title") or "—").replace("|", " ")
        source = (item.get("source") or "—").replace("|", " ")
        path = item.get("path") or "—"
        dl_key = f"btn_{safe_filename(path)}"
        link_cell = "—"
        if show_buttons and path and Path(path).exists():
            with st.container():
                st.download_button(
                    label=f"⬇️ {Path(path).name}",
                    data=as_bytes(Path(path)),
                    file_name=Path(path).name,
                    mime="application/pdf",
                    key=dl_key,
                )
            link_cell = f"`{Path(path).name}`"
        rows_md.append(f"| `{doi}` | {title} | {source} | `{path}` | {link_cell} |")

    # Failures
    for item in results.get("failed", []):
        doi = item.get("doi") or "—"
        title = (item.get("title") or "—").replace("|", " ")
        source = (item.get("source") or "—").replace("|", " ")
        reason = item.get("reason") or "—"
        rows_md.append(f"| `{doi}` | {title} | {source} | — | {reason} |")

    st.markdown("\n".join(rows_md))


# ---------------- Slash Commands ----------------

def handle_slash_command(user_text: str):
    cmd, arg = parse_command(user_text)

    if cmd == "/help":
        help_text = (
            "### Slash commands\n\n"
            "/report `<topic>` — Generate a long-form review.\n\n"
            "/export `[title]` — Export the last (or named) report to DOCX.\n\n"
            "/dois `[title]` — List DOIs found in the last (or named) report.\n\n"
            "/papers `[title]` — Fetch OA PDFs for the last (or named) report via Unpaywall.\n\n"
            "/pdfs `<doi1>, <doi2> ...` — Fetch OA PDFs for explicit DOIs (no report needed).\n\n"
            "/ingest — Ingest downloaded PDFs into the vector store."
        )
        st.markdown(help_text)
        return "Slash commands shown."

    if cmd == "/report":
        topic = arg.strip()
        if not topic:
            return "Usage: `/report <topic>`"
        logger.info(f"/report topic: {topic}")

        report = generate_report_with_specialist(topic)

        if "reports" not in st.session_state:
            st.session_state["reports"] = {}
        title = report.get("title") or topic
        st.session_state["reports"][title] = report
        st.session_state["last_report_title"] = title

        # Auto-export DOCX
        docx_path = REPORTS_DIR / f"{safe_filename(title)}.docx"
        save_report_to_docx(report, str(docx_path))
        st.success(f"Report generated: **{title}**")
        st.download_button(
            "⬇️ Download DOCX",
            data=as_bytes(docx_path),
            file_name=docx_path.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"dl_{safe_filename(title)}",
        )

        return f"(Report generated for: {title})"

    if cmd == "/export":
        reports = st.session_state.get("reports", {})
        title = arg.strip() or st.session_state.get("last_report_title")
        if not title or title not in reports:
            return "No report selected. Use `/export <report title>` or run `/report <topic>` first."
        docx_path = REPORTS_DIR / f"{safe_filename(title)}.docx"
        save_report_to_docx(reports[title], str(docx_path))
        st.success(f"Exported **{title}** to DOCX.")
        st.download_button(
            "⬇️ Download DOCX",
            data=as_bytes(docx_path),
            file_name=docx_path.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"export_{safe_filename(title)}",
        )
        return f"(Exported to DOCX: {docx_path.name})"

    if cmd == "/dois":
        reports = st.session_state.get("reports", {})
        title = arg.strip() or st.session_state.get("last_report_title")
        if not title or title not in reports:
            return "No report to scan. Use `/report <topic>` first."
        report = reports[title]
        dois = list_dois_from_report(report)
        st.session_state["last_dois"] = dois
        if not dois:
            st.info("No DOIs found in the last report.")
            return "(DOI extraction complete.)"
        st.markdown("**DOIs found:**")
        for d in dois:
            st.write(f"- `{d}`")
        return "(DOI extraction complete.)"

    if cmd == "/papers":
        # Download OA PDFs for last (or named) report
        reports = st.session_state.get("reports", {})
        title = arg.strip() or st.session_state.get("last_report_title")
        if not title or title not in reports:
            return "No report to fetch from. Use `/report <topic>` first."
        email = st.session_state.get("student_email", "").strip()
        if not email:
            return "Please set your student email (sidebar) for Unpaywall first."
        result = fetch_pdfs_for_report(reports[title], email)

        st.subheader("PDF Fetch Status")
        render_pdf_status_table(result, show_buttons=True)

        if result["downloaded"]:
            st.success(f"Downloaded {len(result['downloaded'])} OA PDFs to `downloaded_papers/`")
        else:
            st.info("No OA PDFs downloaded.")

        if result["failed"]:
            with st.expander("Failures / Not OA"):
                for f in result["failed"]:
                    st.write(f"- `{f.get('doi') or f.get('label')}`: {f.get('reason')}")

        st.session_state["last_pdf_status"] = result
        return "(PDF fetch complete.)"

    if cmd == "/pdfs":
        # Explicit DOIs: /pdfs doi1, doi2, ...
        email = st.session_state.get("student_email", "").strip()
        if not email:
            return "Please set your student email (sidebar) for Unpaywall first."
        raw = arg.strip()
        if not raw:
            return "Usage: `/pdfs <doi1>, <doi2>, ...`"
        dois = [normalize_doi(t) for t in re.split(r"[,\s]+", raw) if t.strip()]
        result = fetch_pdfs_for_dois(dois, email)

        st.subheader("PDF Fetch Status (Explicit DOIs)")
        render_pdf_status_table(result, show_buttons=True)

        if result["downloaded"]:
            st.success(f"Downloaded {len(result['downloaded'])} OA PDFs to `downloaded_papers/`")
        else:
            st.info("No OA PDFs downloaded.")

        if result["failed"]:
            with st.expander("Failures / Not OA"):
                for f in result["failed"]:
                    st.write(f"- `{f.get('doi') or f.get('label')}`: {f.get('reason')}")

        st.session_state["last_pdf_status"] = result
        return "(PDF fetch complete.)"

    if cmd == "/ingest":
        # Ingest all PDFs in downloaded_papers
        pdfs = [str(x) for x in PAPERS_DIR.glob("*.pdf")]
        if not pdfs:
            return "No PDFs found in `downloaded_papers/`. Run `/papers` or `/pdfs` first."
        res = ingest_pdfs_to_vector_store(pdfs)
        if res.get("ok"):
            st.success(f"Ingested chunks: {res.get('added_chunks', 0)}")
            errs = res.get("errors", [])
            if errs:
                with st.expander("Ingestion warnings"):
                    for e in errs:
                        st.write(f"- {e['path']}: {e['reason']}")
            return "(Ingestion complete.)"
        else:
            return f"Ingestion failed: {res.get('reason','unknown')}"

    return "Unknown command. Try `/help`."


# ---------------- UI ----------------

st.title("🧠 CNfA Research Co-Pilot")
st.caption("Use `/report <topic>` to generate a long-form review, `/export` to download, `/dois` to list DOIs, `/papers` to fetch OA PDFs, `/pdfs` for explicit DOIs, `/ingest` to add them to your KB.")

# Sidebar
with st.sidebar:
    st.header("System Status & Controls")

    # Router/Agent (optional)
    agent_executor = None
    if router_available:
        try:
            agent_executor = get_agent()
            st.success("✅ Agent Router ready.")
        except Exception as e:
            st.warning(f"Agent Router unavailable: {e}")
            agent_executor = None
    else:
        st.info("Agent Router not loaded (fallback mode).")

    # Vector store status
    try:
        client = get_persistent_client()
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        db_doc_count = collection.count()
        st.metric(label="Vector Store Docs", value=db_doc_count)
        if db_doc_count == 0:
            st.warning("Vector store is empty. Please run ingestion first.")
    except Exception as e:
        st.warning(f"⚠️ Vector store not available: {e}")

    # Graph (optional)
    graph_chain = get_graph_chain_or_none()
    if graph_chain:
        try:
            _ = graph_chain.invoke("MATCH (n) RETURN count(n)")
            st.success("✅ Knowledge Graph connected.")
        except Exception:
            st.warning("⚠️ Knowledge Graph not responding.")
    else:
        st.info("Knowledge Graph: not connected (optional).")

    st.markdown("---")
    # Student email for Unpaywall (OA PDFs)
    st.subheader("Open Access Fetch (Unpaywall)")
    email = st.text_input("Student email (required for Unpaywall)", value=st.session_state.get("student_email", ""))
    if email != st.session_state.get("student_email"):
        st.session_state["student_email"] = email

    if st.button("Fetch PDFs for last report (/papers)"):
        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "user", "content": "/papers"})
        st.rerun()

    if st.button("Ingest downloaded PDFs (/ingest)"):
        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "user", "content": "/ingest"})
        st.rerun()

    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        st.rerun()

    st.markdown("---")
    st.header("Reports")
    reports = st.session_state.get("reports", {})
    if reports:
        last_title = st.session_state.get("last_report_title")
        titles = list(reports.keys())
        idx = titles.index(last_title) if last_title in titles else 0
        sel = st.selectbox("Select report", options=titles, index=idx)
        current = reports[sel]

        st.write("**Preview:**")
        st.caption(current.get("title", ""))
        st.markdown(f"- Intro words: {len((current.get('introduction') or '').split())}")
        st.markdown(f"- Theory sections: {len(current.get('theoretical_foundations') or [])}")

        docx_path = REPORTS_DIR / f"{safe_filename(sel)}.docx"
        if docx_path.exists():
            st.download_button("⬇️ Download DOCX",
                               data=as_bytes(docx_path),
                               file_name=docx_path.name,
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               key=f"side_{safe_filename(sel)}")
        else:
            if st.button("Export selected to DOCX"):
                save_report_to_docx(current, str(docx_path))
                st.rerun()
    else:
        st.info("No reports yet. Generate one with `/report <topic>`.")

# Chat area
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I am the CNfA Research Co-Pilot. You can use `/help`, `/report <topic>`, `/export`, `/dois`, `/papers`, `/pdfs`, `/ingest`."
    }]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question… or type /report <topic>")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if is_slash_command(prompt):
            try:
                out = handle_slash_command(prompt)
            except Exception as e:
                out = f"Error: {e}"
            st.markdown(out)
            st.session_state.messages.append({"role": "assistant", "content": out})
        else:
            # Normal agent Q&A (only if router is available)
            if not router_available or get_agent() is None:
                msg = ("Router unavailable. "
                       "Use `/report <topic>` to generate a full review, "
                       "then `/dois` -> `/papers` -> `/ingest`.")
                st.info(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                try:
                    logger.info(f"User query: '{prompt}'")
                    response = get_agent().invoke(
                        {"question": prompt},
                        config={"configurable": {"session_id": st.session_state["session_id"]}}
                    )
                    full_response = response.get("output") or response.get("result") or str(response)
                    st.markdown(full_response)
                except Exception as e:
                    full_response = f"An error occurred: {e}"
                    logger.error(f"Error invoking agent: {e}", exc_info=True)
                    st.error(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
