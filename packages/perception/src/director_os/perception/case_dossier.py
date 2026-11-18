"""Unified multi-source case extractor + CaseDossier synthesis (M5).

Every attached source — the video's own source page, news/court URLs, PDFs
(FOIA / court records), and pasted notes — normalizes into clean text tagged
with `source_id` + `citation`, then synthesizes (with the LLM routing chain
when available, heuristics otherwise) into one `CaseDossier` the Stage 6
Director reads. Authority & conflict priority when sources disagree:

    User Pasted Text/Notes  >  Official Police/Court PDF
                            >  News/Court URL  >  Video Page Summary

with dates attached to every claim and a mechanical ban on invented verdicts
(`verdict_outcome` must appear verbatim-ish in some source text or it is
dropped). FOIA reality check: scanned PDF pages (< 50 extracted chars) fall
back to the vision LLM (gemini-3.5-flash-lite by default) instead of
returning empty text.

Network I/O is injectable (`fetch=`) so tests stay hermetic; PyMuPDF,
trafilatura and BeautifulSoup are optional — each degrades with a clear
error instead of an import crash.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from director_os.core.models import (
    CaseDossier,
    CaseSource,
    DossierFact,
    DossierSourceRef,
)

BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36 DirectorOS-CaseIngest/1.0"
)
FETCH_TIMEOUT_S = 30.0
PDF_MIN_CHARS = 50  # below this a PDF page is treated as a scanned image
MAX_PDF_PAGES = 60
MAX_HOP1_LINKS = 6
MAX_SOURCE_CHARS = 12_000  # per source, fed to synthesis

FetchFn = Callable[[str], Any]  # url -> httpx.Response-like (status_code, text, headers, content)


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------- fetching
def default_fetch(url: str):
    import httpx

    r = httpx.get(
        url,
        headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/pdf,*/*"},
        timeout=FETCH_TIMEOUT_S,
        follow_redirects=True,
    )
    return r


# --------------------------------------------------------------------- PDF path
def _gemini_ocr_page(image_bytes: bytes, api_key: str, model: str) -> str:
    """Read one scanned page image with the vision LLM (inline_data base64)."""
    import httpx

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                    {
                        "text": (
                            "Transcribe all text on this page exactly. "
                            "Output ONLY the transcription."
                        )
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
    }
    base = "https://generativelanguage.googleapis.com"
    url = f"{base}/v1beta/models/{model}:generateContent"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers={"x-goog-api-key": api_key}, json=payload)
        if r.status_code == 401 and "ACCESS_TOKEN_TYPE_UNSUPPORTED" in r.text:
            r = client.post(url, params={"key": api_key}, json=payload)
    if r.status_code != 200:
        raise ExtractionError(f"vision OCR {r.status_code}: {r.text[:200]}")
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(str(p.get("text", "")) for p in parts).strip()


def extract_pdf(
    data: bytes,
    *,
    vision_api_key: str = "",
    vision_model: str = "gemini-3.5-flash-lite",
    log: Callable[[str], None] = lambda _m: None,
) -> str:
    """PyMuPDF text page-by-page; scanned pages (< 50 chars) go to vision OCR."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover - optional dep
        raise ExtractionError("PyMuPDF not installed (uv add pymupdf)") from e

    out: list[str] = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                if i >= MAX_PDF_PAGES:
                    log(f"  [dossier] pdf truncated at {MAX_PDF_PAGES} pages")
                    break
                text = page.get_text("text").strip()
                if len(text) < PDF_MIN_CHARS:
                    # scanned image page -> vision OCR (never return empty text)
                    if not vision_api_key:
                        out.append(f"[page {i + 1}: scanned image, VISION_API_KEY not set]")
                        continue
                    try:
                        pix = page.get_pixmap(dpi=140)
                        ocr = _gemini_ocr_page(pix.tobytes("png"), vision_api_key, vision_model)
                        out.append(ocr if ocr else f"[page {i + 1}: no text detected]")
                    except Exception as e:  # OCR failure must not kill the source
                        out.append(f"[page {i + 1}: OCR failed: {e}]")
                else:
                    out.append(text)
    except Exception as e:
        raise ExtractionError(f"pdf unreadable: {e}") from e
    return "\n\n".join(out).strip()


# --------------------------------------------------------------------- HTML path
_STRIP_TAGS = ("script", "style", "nav", "footer", "aside", "header", "form", "noscript")


def _readable_html(html: str) -> tuple[str, list[str]]:
    """Article body + outbound links via trafilatura, else BeautifulSoup."""
    links: list[str] = []
    try:
        import trafilatura

        text = trafilatura.extract(html, include_links=False, include_tables=True) or ""
    except ImportError:
        text = ""

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        links.append(str(a["href"]))
    if text:
        return text, links

    # readability fallback: strip chrome, take the densest text container
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.find("body") or soup
    for candidate in root.find_all(["div", "section", "article"]):
        if candidate.find(["nav", "aside"]):
            continue
    paragraphs = [p.get_text(" ", strip=True) for p in root.find_all(["p", "li"])]
    body = "\n".join(p for p in paragraphs if len(p) > 40)
    return body, links


def _absolute(base_url: str, href: str) -> str | None:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        from urllib.parse import urlsplit

        parts = urlsplit(base_url)
        return f"{parts.scheme}://{parts.netloc}{href}"
    return None


def extract_url_text(
    url: str,
    *,
    fetch: FetchFn = default_fetch,
    vision_api_key: str = "",
    vision_model: str = "gemini-3.5-flash-lite",
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[str, list[str]]:
    """Fetch one URL; PDF content-types route to the PDF extractor.

    Returns (text, discovered) where discovered carries Hop-1 outbound links
    (only used by the source_page kind).
    """
    try:
        r = fetch(url)
    except Exception as e:
        raise ExtractionError(f"fetch failed: {e}") from e
    if r.status_code != 200:
        raise ExtractionError(f"HTTP {r.status_code} fetching {url[:120]}")
    ctype = str(r.headers.get("content-type", "")).lower()
    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        data = getattr(r, "content", None) or b""
        if not data:
            data = r.text.encode("utf-8", errors="ignore")  # test doubles
        return extract_pdf(
            data, vision_api_key=vision_api_key, vision_model=vision_model, log=log
        ), []
    html = r.text if isinstance(r.text, str) else r.content.decode("utf-8", errors="ignore")
    text, links = _readable_html(html)
    if not text:
        raise ExtractionError(f"no article body extracted from {url[:120]}")
    return text, links


def extract_source_page(
    url: str,
    *,
    fetch: FetchFn = default_fetch,
    vision_api_key: str = "",
    vision_model: str = "gemini-3.5-flash-lite",
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[str, list[dict]]:
    """Video's own page: description + Hop-1 -> Hop-2 linked docs.

    Hop 1: direct .pdf links and same-domain article links on the page
    (Wikimedia file description, COPA case pages). Hop 2 happens in the
    orchestrator: each discovered URL becomes its own `url`/`pdf` source.
    """
    text, links = extract_url_text(
        url, fetch=fetch, vision_api_key=vision_api_key, vision_model=vision_model, log=log
    )
    discovered: list[dict] = []
    seen: set[str] = {url}
    for href in links:
        absu = _absolute(url, href)
        if not absu or absu in seen:
            continue
        seen.add(absu)
        is_pdf = absu.lower().endswith(".pdf")
        is_doc = is_pdf or re.search(r"/(wiki|commons|case|doc|file|media)\b", absu) is not None
        if not is_doc:
            continue
        if any(d["url"] == absu for d in discovered):
            continue
        discovered.append(
            {
                "kind": "pdf" if is_pdf else "url",
                "label": f"linked: {Path(absu).name[:80]}",
                "url": absu,
            }
        )
        if len(discovered) >= MAX_HOP1_LINKS:
            break
    return text, discovered


def extract_case_source(
    source: CaseSource,
    *,
    fetch: FetchFn = default_fetch,
    vision_api_key: str = "",
    vision_model: str = "gemini-3.5-flash-lite",
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[str, list[dict]]:
    """Normalize one CaseSource -> (extracted_text, discovered_children).

    Raises ExtractionError with a human-readable cause (persisted to
    `case_sources.error` by the orchestrator).
    """
    if source.kind == "text":
        body = source.text.strip()
        if not body:
            raise ExtractionError("pasted text is empty")
        return body, []
    if source.kind == "pdf":
        p = Path(source.uri_or_storage_key)
        if not p.exists():
            raise ExtractionError(f"uploaded file missing: {p.name}")
        return (
            extract_pdf(
                p.read_bytes(), vision_api_key=vision_api_key, vision_model=vision_model, log=log
            ),
            [],
        )
    if not source.uri_or_storage_key:
        raise ExtractionError(f"{source.kind} source has no url")
    if source.kind == "source_page":
        return extract_source_page(
            source.uri_or_storage_key,
            fetch=fetch,
            vision_api_key=vision_api_key,
            vision_model=vision_model,
            log=log,
        )
    text, _links = extract_url_text(
        source.uri_or_storage_key,
        fetch=fetch,
        vision_api_key=vision_api_key,
        vision_model=vision_model,
        log=log,
    )
    return text, []


# --------------------------------------------------------------------- synthesis
AUTHORITY = {"text": 4, "pdf": 3, "url": 2, "source_page": 1}

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_RE_DATE = re.compile(rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}")
_RE_AGE = re.compile(r"\b(\d{2})[- ]year[- ]old\b", re.I)
_RE_CHARGE = re.compile(
    r"(?:charged with|arrested(?: on suspicion of)?|booked on|cited for)\s+([^.]{4,100})", re.I
)
_RE_TEST = re.compile(
    r"((?:breath|blood|urine)\s+test[^.]*?\d(?:\.\d+)?\s*(?:%|percent|g/dL|grams per 100mL)?[^.]*\."
    r"|[^.]*?\b\d(?:\.\d+)?\s*(?:%|percent|g/dL|grams per 100mL)[^.]*\.)",
    re.I,
)
_RE_VERDICT = re.compile(
    r"([^.]*(?:pleaded|pled)\s+(?:guilty|no contest)[^.]*\."
    r"|[^.]*found\s+guilty[^.]*\."
    r"|[^.]*sentenced to[^.]*\.)",
    re.I,
)
_SIGNAL_WORDS = (
    "bodycam",
    "body cam",
    "dashcam",
    "arrest",
    "dui",
    "bac",
    "miranda",
    "breath",
    "blood",
    "refus",
    "license",
    "susp",
    "probation",
    "resign",
    "fired",
    "terminated",
    "patrol",
    "on duty",
    "dash camera",
    "police",
    "officer",
    "sheriff",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _source_texts(sources: list[CaseSource]) -> list[tuple[CaseSource, str]]:
    """Ready sources only, highest authority first (stable within a tier)."""
    ready = [
        (s, s.extracted_text.strip())
        for s in sources
        if s.status == "ready" and s.extracted_text.strip()
    ]
    ready.sort(key=lambda p: (-AUTHORITY.get(p[0].kind, 0), p[0].created_at or ""))
    return ready


def _citation(source: CaseSource) -> str:
    if source.kind == "text":
        return source.label or "pasted notes"
    if source.uri_or_storage_key:
        return source.uri_or_storage_key[:160]
    return source.label or source.kind


def heuristic_dossier(
    sources: list[CaseSource], stage4_ocr: dict[str, Any] | None = None
) -> CaseDossier:
    """No-LLM synthesis: regex facts, authority-merged, citation-attributed."""
    dossier = CaseDossier(
        sources=[
            DossierSourceRef(
                source_id=s.id, kind=s.kind, label=s.label or _citation(s), citation=_citation(s)
            )
            for s in sources
            if s.status == "ready"
        ]
    )
    ready = _source_texts(sources)
    for source, text in ready:
        if dossier.subject_name is None:
            m = re.search(
                r"\b(?:Officer|Sgt\.?|Sergeant|Deputy|Corporal|Detective)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                text,
            )
            if m:
                dossier.subject_name = m.group(1)
        if dossier.age is None:
            m = _RE_AGE.search(text)
            if m:
                dossier.age = int(m.group(1))
        if dossier.incident_datetime is None:
            m = _RE_DATE.search(text)
            if m:
                dossier.incident_datetime = m.group(0)
        if dossier.location is None:
            m = re.search(
                r"\bin\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s([A-Z]{2}|[A-Z][a-z]+)\b", text
            )
            if m:
                dossier.location = f"{m.group(1)}, {m.group(2)}"
        if dossier.verdict_outcome is None:
            m = _RE_VERDICT.search(text)
            if m:
                verdict = m.group(1).strip()
                # a plea is usually followed by a sentencing sentence — keep it
                tail = text[m.end() : m.end() + 220]
                m2 = re.match(r"\s*([^.]*sentenced[^.]*\.)", tail, re.I)
                if m2:
                    verdict = f"{verdict} {m2.group(1).strip()}"
                dossier.verdict_outcome = verdict[:300]
        for m in _RE_CHARGE.finditer(text):
            charge = m.group(1).strip().rstrip(".,;")[:120]
            if charge.lower() not in [c.lower() for c in dossier.charges]:
                dossier.charges.append(charge)
            if len(dossier.charges) >= 6:
                break
        for m in _RE_TEST.finditer(text):
            tr = m.group(1).strip()[:200]
            if tr.lower() not in [t.lower() for t in dossier.test_results]:
                dossier.test_results.append(tr)
            if len(dossier.test_results) >= 6:
                break
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if len(dossier.key_facts) >= 12:
                break
            s = sent.strip()
            if len(s) < 30 or len(s) > 300:
                continue
            if not any(w in _norm(s) for w in _SIGNAL_WORDS):
                continue
            if any(f.fact.lower() == s.lower() for f in dossier.key_facts):
                continue
            m = _RE_DATE.search(s)
            dossier.key_facts.append(
                DossierFact(
                    fact=s,
                    date=m.group(0) if m else None,
                    source_id=source.id,
                    citation=_citation(source),
                )
            )

    if stage4_ocr:
        ocr_text = " ".join(str(v) for v in stage4_ocr.values())[:600]
        if ocr_text.strip():
            dossier.key_facts.append(
                DossierFact(
                    fact=f"Axon/bodycam overlay OCR: {ocr_text[:280]}",
                    date=None,
                    source_id="stage4_ocr",
                    citation="raw video overlay OCR",
                )
            )
    return dossier


def _dossier_prompt(ready: list[tuple[CaseSource, str]], stage4_ocr: dict[str, Any] | None) -> str:
    blocks = []
    for source, text in ready:
        blocks.append(
            f"### SOURCE {source.id} (kind={source.kind}, label={source.label or '-'})\n"
            f"{text[:MAX_SOURCE_CHARS]}"
        )
    ocr = ""
    if stage4_ocr:
        ocr = (
            "\n### SOURCE stage4_ocr (raw video overlay OCR)\n"
            + " ".join(str(v) for v in stage4_ocr.values())[:1500]
        )
    return (
        "Extract a case dossier from these sources. Copy facts — never invent names, "
        "charges, test results, or verdicts. Every key_fact MUST cite a real source id "
        "from the headers. `verdict_outcome` ONLY if a source explicitly states a "
        "conviction/plea/sentence; otherwise null. When sources disagree, prefer "
        "pasted text > pdf > url > source_page and keep the newer date.\n\n"
        + "\n\n".join(blocks)
        + ocr
        + '\n\nReturn ONLY JSON: {"subject_name": str|null, "age": int|null, '
        '"role_or_agency": str|null, "incident_datetime": str|null, "location": str|null, '
        '"inciting_reason": str|null, "key_facts": [{"fact": str, "date": str|null, '
        '"source_id": str, "citation": str}], "charges": [str], "test_results": [str], '
        '"verdict_outcome": str|null}'
    )


def _strip_fences(text: str) -> str:
    """Lenient JSON extraction: drop ```json fences, keep the outermost object."""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        return text[first : last + 1]
    return text


def llm_dossier(
    sources: list[CaseSource],
    stage4_ocr: dict[str, Any] | None,
    complete_json: Callable[[str, str], str],
) -> CaseDossier | None:
    """LLM synthesis through an injected complete_json(system, user) callable."""
    import json as _json

    ready = _source_texts(sources)
    if not ready:
        return None
    try:
        raw = complete_json(
            "You are a meticulous true-crime case researcher. You never invent facts.",
            _dossier_prompt(ready, stage4_ocr),
        )
        data = _json.loads(_strip_fences(raw))
    except Exception:
        return None

    known_ids = {s.id for s, _t in ready}
    dossier = CaseDossier(
        sources=[
            DossierSourceRef(
                source_id=s.id, kind=s.kind, label=s.label or _citation(s), citation=_citation(s)
            )
            for s in sources
            if s.status == "ready"
        ]
    )
    dossier.subject_name = data.get("subject_name") or None
    dossier.age = data.get("age") if isinstance(data.get("age"), int) else None
    dossier.role_or_agency = data.get("role_or_agency") or None
    dossier.incident_datetime = data.get("incident_datetime") or None
    dossier.location = data.get("location") or None
    dossier.inciting_reason = data.get("inciting_reason") or None
    dossier.charges = [str(c)[:160] for c in (data.get("charges") or [])][:8]
    dossier.test_results = [str(t)[:200] for t in (data.get("test_results") or [])][:8]

    ready_texts = " ".join(_norm(t) for _s, t in ready)
    verdict = data.get("verdict_outcome")
    if isinstance(verdict, str) and verdict.strip():
        # mechanical ban on invented verdicts: the claim's distinctive words
        # must trace back to some source text (stop-words like "of" never
        # count — at least 3 meaningful words or half of them must be present)
        meaningful = [w for w in _norm(verdict).split() if len(w) >= 4]
        hits = sum(1 for w in meaningful if w in ready_texts)
        traced = meaningful and hits >= min(3, len(meaningful))
        if traced:
            dossier.verdict_outcome = verdict.strip()[:300]

    for f in (data.get("key_facts") or [])[:16]:
        if not isinstance(f, dict):
            continue
        fact = str(f.get("fact", "")).strip()
        sid = str(f.get("source_id", ""))
        if not fact or sid not in known_ids:
            continue
        dossier.key_facts.append(
            DossierFact(
                fact=fact[:300],
                date=str(f["date"])[:60] if f.get("date") else None,
                source_id=sid,
                citation=str(f.get("citation", ""))[:160]
                or _citation(next(s for s, _t in ready if s.id == sid)),
            )
        )
    return dossier


def synthesize_dossier(
    sources: list[CaseSource],
    stage4_ocr: dict[str, Any] | None = None,
    *,
    complete_json: Callable[[str, str], str] | None = None,
) -> CaseDossier:
    """LLM synthesis when a chain is available, heuristic otherwise."""
    if complete_json is not None:
        dossier = llm_dossier(sources, stage4_ocr, complete_json)
        if dossier is not None and (dossier.key_facts or dossier.verdict_outcome):
            return dossier
    return heuristic_dossier(sources, stage4_ocr)
