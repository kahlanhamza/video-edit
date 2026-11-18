"""Multi-source case extraction + CaseDossier synthesis (M5).

Hermetic: fetching is injected; PDFs are generated in-memory with PyMuPDF;
the vision OCR path is monkeypatched. Authority priority
(text > pdf > url > source_page), mechanical ban on invented verdicts, and
the scanned-PDF reality check are all pinned.
"""

from __future__ import annotations

import pytest
from director_os.core.models import CaseSource
from director_os.perception.case_dossier import (
    ExtractionError,
    extract_case_source,
    extract_pdf,
    extract_source_page,
    heuristic_dossier,
    llm_dossier,
    synthesize_dossier,
)


class FakeResponse:
    def __init__(self, status_code=200, text="", content=b"", headers=None):
        self.status_code = status_code
        self.text = text
        self.content = content
        self.headers = headers or {}


def _pdf_bytes(pages: list[str], scanned: bool = False) -> bytes:
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if not scanned:
            page.insert_text((72, 72), text, fontsize=11)
    return doc.tobytes()


def _make_pdf_response(pages: list[str], scanned: bool = False) -> FakeResponse:
    return FakeResponse(
        content=_pdf_bytes(pages, scanned=scanned), headers={"content-type": "application/pdf"}
    )


# ------------------------------------------------------------------ extractors
def test_extract_text_source_direct():
    src = CaseSource(
        id="s1", project_id="p", kind="text", label="notes", text="officer resigned in 2023"
    )
    text, discovered = extract_case_source(src)
    assert text == "officer resigned in 2023"
    assert discovered == []


def test_extract_text_empty_raises():
    src = CaseSource(id="s1", project_id="p", kind="text", text="  ")
    with pytest.raises(ExtractionError, match="empty"):
        extract_case_source(src)


def test_extract_pdf_with_text_pages():
    src = CaseSource(id="s2", project_id="p", kind="pdf", label="court.pdf", uri_or_storage_key="")
    # write a real pdf to a temp path
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(
            _pdf_bytes(
                ["Court record: defendant pleaded guilty to DUI. Sentenced to 12 months probation."]
            )
        )
        path = Path(f.name)
    src = src.model_copy(update={"uri_or_storage_key": str(path)})
    text, _ = extract_case_source(src)
    assert "pleaded guilty" in text.replace("\n", " ")


def test_extract_pdf_scanned_pages_fall_back_to_vision(monkeypatch):
    monkeypatch.setattr(
        "director_os.perception.case_dossier._gemini_ocr_page",
        lambda img, key, model: "Scanned court page: guilty plea, 12 months probation.",
    )
    data = _pdf_bytes(["", ""], scanned=True)  # no extractable text
    text = extract_pdf(data, vision_api_key="AQ.key", vision_model="gemini-3.5-flash-lite")
    assert "Scanned court page" in text
    assert "VISION_API_KEY not set" not in text


def test_extract_pdf_scanned_without_key_notes_it():
    data = _pdf_bytes([""], scanned=True)
    text = extract_pdf(data)  # no key -> explicit note, never silent empty
    assert "[page 1: scanned image, VISION_API_KEY not set]" in text


def test_extract_url_routes_pdf_content_type():
    src = CaseSource(
        id="s3", project_id="p", kind="url", uri_or_storage_key="https://copa.example/case.pdf"
    )
    pdf = _pdf_bytes(["Case 2022-074: officer arrested for DUI while on duty."])
    fetch = lambda url: FakeResponse(content=pdf, headers={"content-type": "application/pdf"})  # noqa: E731
    text, _ = extract_case_source(src, fetch=fetch)
    assert "officer arrested for DUI" in text.replace("\n", " ")


def test_extract_url_html_body_and_error_paths():
    html = (
        "<html><body><nav>menu</nav><article><p>"
        + "Eustis officer sentenced. " * 12
        + "</p></article></body></html>"
    )
    src = CaseSource(
        id="s4", project_id="p", kind="url", uri_or_storage_key="https://news.example/story"
    )
    text, _ = extract_case_source(src, fetch=lambda url: FakeResponse(text=html))
    assert "Eustis officer sentenced" in text

    src404 = src.model_copy(update={"id": "s5"})
    with pytest.raises(ExtractionError, match="HTTP 404"):
        extract_case_source(src404, fetch=lambda url: FakeResponse(status_code=404))


def test_source_page_discovers_linked_docs_hop1():
    html = """
    <html><body>
      <p>Bodycam: Drunk Cop Arrested While On Duty. Eustis, FL. July 04, 2022.</p>
      <a href="https://files.example/report.pdf">full report</a>
      <a href="/wiki/Bodycam_Eustis">wiki article</a>
      <a href="https://ads.example/track">ad</a>
    </body></html>
    """
    fetch = lambda url: FakeResponse(text=html)  # noqa: E731
    text, discovered = extract_source_page("https://commons.example/file", fetch=fetch)
    assert "Eustis" in text
    urls = [d["url"] for d in discovered]
    assert "https://files.example/report.pdf" in urls
    assert "https://commons.example/wiki/Bodycam_Eustis" in urls
    assert "https://ads.example/track" not in urls
    kinds = {d["url"]: d["kind"] for d in discovered}
    assert kinds["https://files.example/report.pdf"] == "pdf"


# ------------------------------------------------------------------ synthesis
def _source(sid: str, kind: str, text: str, label: str = "", created: str = "") -> CaseSource:
    return CaseSource(
        id=sid,
        project_id="p",
        kind=kind,
        label=label,
        extracted_text=text,
        status="ready",
        created_at=created,  # type: ignore[arg-type]
    )


def test_heuristic_priority_text_beats_url_on_verdict():
    sources = [
        _source("u1", "url", "The July 4, 2022 arrest was widely reported. The case was pending."),
        _source(
            "t1", "text", "Update: he pleaded guilty. Sentenced to 12 months probation.", "my notes"
        ),
    ]
    d = heuristic_dossier(sources)
    assert d.verdict_outcome is not None
    assert "12 months probation" in d.verdict_outcome
    assert d.incident_datetime == "July 4, 2022"


def test_heuristic_extracts_charges_and_facts_with_citations():
    sources = [
        _source(
            "p1",
            "pdf",
            "Officer Miguel Irizarry, a 46-year-old veteran, was arrested and charged with DUI. "
            "A breath test showed 0.21 percent. "
            "The incident happened in Eustis, Florida on July 4, 2022.",
            "court.pdf",
        )
    ]
    d = heuristic_dossier(sources)
    assert d.subject_name == "Miguel Irizarry"
    assert d.age == 46
    assert d.location and "Eustis" in d.location
    assert any("DUI" in c for c in d.charges)
    assert any("0.21" in t for t in d.test_results)
    assert d.key_facts and all(f.source_id == "p1" for f in d.key_facts)
    assert d.incident_datetime is not None


def test_heuristic_no_verdict_when_none_stated():
    d = heuristic_dossier(
        [_source("u1", "url", "The officer was arrested on suspicion of DUI in 2022.")]
    )
    assert d.verdict_outcome is None  # never invent outcomes


def test_synthesize_dossier_prefers_llm_but_falls_back():
    good_json = (
        '{"subject_name": "Miguel Irizarry", "age": 46, '
        '"key_facts": [{"fact": "arrested on duty", "source_id": "p1", '
        '"citation": "court.pdf"}], "charges": ["DUI"], "test_results": [], '
        '"verdict_outcome": null}'
    )
    sources = [_source("p1", "pdf", "Officer Miguel Irizarry arrested for DUI while on duty.")]
    d = synthesize_dossier(sources, None, complete_json=lambda s, u: good_json)
    assert d.subject_name == "Miguel Irizarry"
    assert d.key_facts[0].source_id == "p1"


def test_llm_dossier_drops_invented_verdicts_and_unknown_citations():
    lying_json = (
        '{"subject_name": null, "key_facts": ['
        '{"fact": "was convicted and jailed", "source_id": "made_up", "citation": "x"}],'
        ' "charges": [], "test_results": [], '
        '"verdict_outcome": "Convicted of 47 felonies and jailed for life"}'
    )
    sources = [_source("p1", "url", "The officer was arrested on suspicion of DUI. Case pending.")]
    d = llm_dossier(sources, None, lambda s, u: lying_json)
    assert d.verdict_outcome is None, "verdict must trace back to source text"
    assert d.key_facts == [], "facts citing unknown sources are dropped"


def test_llm_dossier_keeps_verdict_that_traces_to_text():
    honest_json = (
        '{"subject_name": null, "key_facts": [], "charges": [], "test_results": [], '
        '"verdict_outcome": "He pleaded guilty and was sentenced to 12 months probation."}'
    )
    sources = [
        _source("t1", "text", "Update: he pleaded guilty and was sentenced to 12 months probation.")
    ]
    d = llm_dossier(sources, None, lambda s, u: honest_json)
    assert d.verdict_outcome is not None and "12 months probation" in d.verdict_outcome


def test_stage4_ocr_joins_dossier():
    d = heuristic_dossier([], {"overlay": "AXON 7 BWC 2022-07-04 21:41 OFF IRIZARRY"})
    assert any("AXON" in f.fact for f in d.key_facts)
    assert d.key_facts[-1].source_id == "stage4_ocr"
