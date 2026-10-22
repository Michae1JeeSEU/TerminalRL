from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_QA = ROOT / "qa" / "pdf"
XLSX_QA = ROOT / "qa" / "xlsx"
sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    integrity = load_json(PDF_QA / "integrity_report.json")
    visual = load_json(PDF_QA / "visual_report.json")
    manifest = load_json(XLSX_QA / "build_manifest.json")
    zip_verify = load_json(XLSX_QA / "zip_verify.json")
    updated_visibility = load_json(XLSX_QA / "updated_row_visibility.json")

    pdfs = {}
    for item in integrity:
        output = Path(item["output"])
        stem = output.stem
        page_dir = PDF_QA / stem
        rendered_pages = sorted(page_dir.glob("page-*.png"))
        pdfs[stem] = {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "pages": item["pages"],
            "rendered_pages": len(rendered_pages),
            "text_integrity": item["text_integrity"],
            "missing_source_lines": item["missing_source_lines"],
            "missing_reference_ids": item["missing_reference_ids"],
            "fonts": item["fonts"],
            "visual_contact_sheet": visual[stem],
        }

    workbook = ROOT / "04_literature_review.xlsx"
    result = {
        "pdfs": pdfs,
        "xlsx": {
            "path": str(workbook),
            "size_bytes": workbook.stat().st_size,
            "record_count": manifest["record_count"],
            "full_text_count": manifest["full_text_count"],
            "abstract_only_count": manifest["abstract_only_count"],
            "abstract_only_ids": manifest["abstract_only_ids"],
            "local_main_pdf_count": manifest["local_main_pdf_count"],
            "missing_sheet_row_count": manifest["missing_sheet_row_count"],
            "html_read_without_local_file_count": manifest["html_read_without_local_file_count"],
            "zip_verify": zip_verify,
            "updated_row_visibility": updated_visibility,
            "preview_sizes_bytes": {
                path.name: path.stat().st_size
                for path in sorted(XLSX_QA.glob("*.png"))
            },
        },
    }
    output = ROOT / "qa" / "qa_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
