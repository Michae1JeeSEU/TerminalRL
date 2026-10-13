from __future__ import annotations

import argparse
import json
import re
import sys
from html import escape
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
QA = ROOT / "qa" / "pdf"

BODY_FONT = "LowAltBody"
BOLD_FONT = "LowAltHeading"


def choose_font(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No usable Chinese font found: " + ", ".join(map(str, candidates)))


def register_fonts() -> dict[str, str]:
    body_path = choose_font(
        [
            Path(r"C:\Windows\Fonts\STSONG.TTF"),
            Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
            Path(r"C:\Windows\Fonts\Deng.ttf"),
        ]
    )
    heading_path = choose_font(
        [
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simsunb.ttf"),
            body_path,
        ]
    )
    pdfmetrics.registerFont(TTFont(BODY_FONT, str(body_path)))
    if heading_path == body_path:
        heading_font = BODY_FONT
    else:
        pdfmetrics.registerFont(TTFont(BOLD_FONT, str(heading_path)))
        heading_font = BOLD_FONT
    return {"body": BODY_FONT, "heading": heading_font, "body_path": str(body_path), "heading_path": str(heading_path)}


def load_documents() -> dict[str, dict]:
    documents: dict[str, dict] = {}
    for filename in ("track_A.json", "track_B.json", "track_C.json"):
        data = json.loads((EVIDENCE / filename).read_text(encoding="utf-8"))
        entries = data["documents"] if isinstance(data, dict) else data
        for entry in entries:
            documents[entry["id"]] = entry
    return documents


def reference_text(entry: dict) -> tuple[str, str]:
    authors = "，".join(str(author) for author in entry.get("authors", []))
    year = str(entry.get("year", ""))
    title = str(entry.get("title", ""))
    venue = str(entry.get("venue", ""))
    doi = str(entry.get("doi", ""))
    first = f"[{entry['id']}] {authors}. {year}. {title}. {venue}."
    second = f"DOI: {doi}"
    return first, second


def cited_ids(source: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in re.findall(r"\[([A-C]\d{2})\]", source):
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result


def classify_line(line: str, index: int) -> str:
    stripped = line.strip()
    if index == 0:
        return "title"
    if not stripped:
        return "blank"
    if re.match(r"^\d+\.\d+\.\d+\s+", stripped):
        return "h3"
    if re.match(r"^\d+\.\d+\s+", stripped):
        return "h2"
    if re.match(r"^\d+\s+\S", stripped):
        return "h1"
    if stripped in {"摘要", "英文摘要", "符号及缩略语说明", "框架说明", "参考文献", "附录"}:
        return "h1"
    if stripped == "章节与成果对应":
        return "h2"
    return "body"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self):  # noqa: N802
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):  # noqa: N802
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(page_count)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count: int) -> None:
        self.saveState()
        self.setFont(BODY_FONT, 9.5)
        self.setFillColor(colors.HexColor("#333333"))
        self.drawCentredString(A4[0] / 2, 12 * mm, f"Page {self._pageNumber} / {page_count}")
        self.restoreState()


class ReportDocument(BaseDocTemplate):
    def __init__(self, filename: Path, styles: dict[str, ParagraphStyle]):
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=22 * mm,
            rightMargin=22 * mm,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            title=filename.stem,
            author="",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="main")
        self.addPageTemplates([PageTemplate(id="normal", frames=[frame])])


def make_styles(fonts: dict[str, str]) -> dict[str, ParagraphStyle]:
    common = dict(textColor=colors.black, wordWrap="CJK", splitLongWords=True)
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            **common,
            fontName=fonts["heading"],
            fontSize=17,
            leading=25,
            alignment=TA_CENTER,
            spaceAfter=13,
            keepWithNext=True,
        ),
        "meta": ParagraphStyle(
            "ReportMeta",
            **common,
            fontName=fonts["body"],
            fontSize=10.8,
            leading=17,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "HeadingOne",
            **common,
            fontName=fonts["heading"],
            fontSize=14,
            leading=21,
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "HeadingTwo",
            **common,
            fontName=fonts["heading"],
            fontSize=12.6,
            leading=19,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "HeadingThree",
            **common,
            fontName=fonts["heading"],
            fontSize=11.8,
            leading=18,
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            **common,
            fontName=fonts["body"],
            fontSize=11.4,
            leading=18,
            spaceAfter=5,
            alignment=TA_LEFT,
            allowWidows=0,
            allowOrphans=0,
        ),
        "ref": ParagraphStyle(
            "Reference",
            **common,
            fontName=fonts["body"],
            fontSize=10.6,
            leading=16,
            spaceAfter=3,
            alignment=TA_LEFT,
            allowWidows=0,
            allowOrphans=0,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    # Escape source text so all characters remain literal ReportLab paragraph text.
    return Paragraph(escape(text).replace("\n", "<br/>").replace("  ", "&#160; "), style)


def build_one(source_path: Path, output_path: Path, document_index: int, docs: dict[str, dict]) -> dict:
    source = source_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    fonts = register_fonts()
    styles = make_styles(fonts)
    story = []
    previous_kind = None
    for index, line in enumerate(lines):
        kind = classify_line(line, index)
        if kind == "blank":
            if previous_kind not in {"title", "h1", "h2", "h3"}:
                story.append(Spacer(1, 4.5))
        elif kind == "title":
            story.append(paragraph(line, styles["title"]))
        elif index == 1 and not re.match(r"^\d", line.strip()):
            story.append(paragraph(line, styles["meta"]))
        else:
            story.append(paragraph(line, styles[kind]))
        if kind != "blank":
            previous_kind = kind

    source_ids = cited_ids(source)
    if document_index in (1, 2):
        selected_ids = sorted(docs) if document_index == 1 else sorted(source_ids)
        story.append(Spacer(1, 8))
        story.append(paragraph("参考文献", styles["h1"]))
        for doc_id in selected_ids:
            if doc_id not in docs:
                raise KeyError(f"Citation {doc_id} is absent from track JSON")
            first, second = reference_text(docs[doc_id])
            story.append(KeepTogether([paragraph(first, styles["ref"]), paragraph(second, styles["ref"])]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocument(output_path, styles)
    doc.build(story, canvasmaker=NumberedCanvas)

    reader = PdfReader(str(output_path))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    extracted_without_footer = re.sub(r"Page\s+\d+\s*/\s+\d+", "", extracted)
    normalized_extracted = re.sub(r"\s+", "", extracted_without_footer)
    missing_lines: list[str] = []
    for line in lines:
        stripped = re.sub(r"\s+", "", line.strip())
        if stripped and stripped not in normalized_extracted:
            missing_lines.append(line)
    missing_refs = []
    if document_index in (1, 2):
        selected_ids = sorted(docs) if document_index == 1 else sorted(source_ids)
        for doc_id in selected_ids:
            if f"[{doc_id}]" not in extracted or str(docs[doc_id].get("doi", "")) not in extracted:
                missing_refs.append(doc_id)
    qa = {
        "source": str(source_path),
        "output": str(output_path),
        "pages": len(reader.pages),
        "source_lines": len(lines),
        "source_citation_ids": source_ids,
        "reference_ids": sorted(docs) if document_index == 1 else (sorted(source_ids) if document_index == 2 else []),
        "missing_source_lines": missing_lines,
        "missing_reference_ids": missing_refs,
        "fonts": fonts,
        "text_integrity": not missing_lines and not missing_refs,
    }
    return qa


def main() -> int:
    parser = argparse.ArgumentParser(description="Build three Chinese low-altitude-traffic PDF reports from TXT sources.")
    parser.add_argument("--report", choices=("all", "01", "02", "03"), default="all")
    args = parser.parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    docs = load_documents()
    specs = [
        (1, "01_literature_synthesis.txt", "01_literature_synthesis.pdf"),
        (2, "02_research_proposal.txt", "02_research_proposal.pdf"),
        (3, "03_dissertation_outline.txt", "03_dissertation_outline.pdf"),
    ]
    if args.report != "all":
        specs = [item for item in specs if item[0] == int(args.report)]
    results = []
    for index, source_name, output_name in specs:
        source_path = ROOT / source_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        result = build_one(source_path, ROOT / output_name, index, docs)
        results.append(result)
    report_path = QA / "integrity_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["text_integrity"] for item in results) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
