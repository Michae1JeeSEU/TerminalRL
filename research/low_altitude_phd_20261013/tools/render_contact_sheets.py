from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "qa" / "pdf"


def make_contact_sheet(page_paths: list[Path], output: Path) -> tuple[int, int]:
    thumbnails = []
    for page_path in page_paths:
        with Image.open(page_path) as image:
            image = image.convert("RGB")
            thumb_width = 360
            thumb_height = max(1, round(image.height * thumb_width / image.width))
            thumbnails.append((page_path.name, image.resize((thumb_width, thumb_height))))
    columns = 2
    label_height = 28
    gap = 14
    rows = (len(thumbnails) + columns - 1) // columns
    cell_height = label_height + (thumbnails[0][1].height if thumbnails else 0)
    sheet = Image.new("RGB", (columns * 360 + (columns + 1) * gap, rows * cell_height + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, thumbnail) in enumerate(thumbnails):
        row, column = divmod(index, columns)
        left = gap + column * (360 + gap)
        top = gap + row * cell_height
        draw.text((left, top + 5), name, fill="black")
        sheet.paste(thumbnail, (left, top + label_height))
    sheet.save(output)
    return sheet.width, sheet.height


def main() -> None:
    report = {}
    for stem in ("01_literature_synthesis", "02_research_proposal", "03_dissertation_outline"):
        directory = QA / stem
        pages = sorted(directory.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
        output = directory / "contact_sheet.png"
        width, height = make_contact_sheet(pages, output)
        report[stem] = {
            "page_count": len(pages),
            "contact_sheet": str(output),
            "contact_sheet_size_px": [width, height],
            "page_sizes_px": {path.name: list(Image.open(path).size) for path in pages},
        }
    (QA / "visual_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
