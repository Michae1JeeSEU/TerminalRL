import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const TOOLS_DIR = path.dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = path.resolve(TOOLS_DIR, "..");
const REPO_ROOT = path.resolve(PACKAGE_ROOT, "..", "..");
const EVIDENCE = path.join(PACKAGE_ROOT, "evidence");
const OUTPUT = path.join(PACKAGE_ROOT, "04_literature_review.xlsx");
const QA_DIR = path.join(PACKAGE_ROOT, "qa", "xlsx");
const FONT = "Arial";

const readJson = async (name) => JSON.parse(await fs.readFile(path.join(EVIDENCE, name), "utf8"));

const asText = (value) => {
  if (value === undefined || value === null || value === "") return "";
  if (Array.isArray(value)) {
    return value.map((item) => asText(item)).filter((item) => item !== "").join("\n");
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => `${key}: ${asText(item)}`)
      .filter((item) => !item.endsWith(": "))
      .join("\n");
  }
  return String(value);
};

const firstValue = (record, keys) => {
  for (const key of keys) {
    const value = record?.[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return "";
};

const authorsText = (record) => {
  const raw = firstValue(record, ["authors", "author", "full_authors"]);
  let authors = Array.isArray(raw) ? raw.map(asText).filter(Boolean) : [asText(raw)].filter(Boolean);
  if (authors.length === 0) return "";
  if (!authors[0].includes("★")) authors[0] = `★ ${authors[0]}`;
  return authors.join("; ");
};

const relativeToWorkbook = (value) => {
  const text = asText(value);
  if (!text) return "";
  const workbookRoot = path.win32.resolve(PACKAGE_ROOT.replaceAll("/", "\\"));
  const candidate = text.replaceAll("/", "\\");
  const absolute = path.win32.isAbsolute(candidate)
    ? candidate
    : candidate.toLowerCase().startsWith("research\\")
      ? path.win32.resolve(REPO_ROOT.replaceAll("/", "\\"), candidate)
      : path.win32.resolve(workbookRoot, candidate);
  return path.win32.relative(workbookRoot, absolute).replaceAll("\\", "/");
};

const titleWithYear = (record) => {
  const title = asText(firstValue(record, ["title", "paper_title"]));
  const year = firstValue(record, ["year", "publication_year"]);
  return year === "" ? title : `${title}（${asText(year)}）`;
};

const critiqueText = (critique) => {
  if (!critique || typeof critique !== "object") return { assessment: "", relevance: "", all: "" };
  const labels = { assessment: "本次评述", relevance: "研究启示" };
  const entries = Object.entries(critique)
    .map(([key, value]) => [labels[key] ?? key, asText(value)])
    .filter(([, value]) => value !== "");
  const assessment = asText(critique.assessment);
  const relevance = asText(critique.relevance);
  return {
    assessment,
    relevance,
    all: entries.map(([key, value]) => `${key}：${value}`).join("\n"),
  };
};

const addLabel = (label, value) => {
  const text = asText(value);
  return text ? `${label}：${text}` : "";
};

const joinParts = (...parts) => parts.filter(Boolean).join("\n");

const makeWorkbookRelativeData = (records) => records.map((record) => ({
  ...record,
  id: asText(firstValue(record, ["id", "ID", "paper_id"])),
  title: asText(firstValue(record, ["title", "paper_title"])),
  authors: authorsText(record),
  year: firstValue(record, ["year", "publication_year"]),
  venue: asText(firstValue(record, ["venue", "journal", "conference"])),
  doi: asText(firstValue(record, ["doi", "DOI"])),
  primary_url: asText(firstValue(record, ["primary_url", "url", "source_url"])),
  fulltext_url: asText(firstValue(record, ["fulltext_url", "full_text_url", "pdf_url"])),
  local_pdf_relative: relativeToWorkbook(firstValue(record, ["local_pdf", "pdf_path"])),
  local_text_relative: relativeToWorkbook(firstValue(record, ["local_text", "text_path"])),
  reading_status: asText(firstValue(record, ["reading_status", "status"])),
  version: asText(firstValue(record, ["version", "document_version"])),
  method: asText(firstValue(record, ["method", "methods"])),
  scenario: asText(firstValue(record, ["scenario", "setting"])),
  findings: asText(firstValue(record, ["findings", "conclusions", "main_findings"])),
  quantitative_findings: asText(firstValue(record, ["quantitative_findings", "quantitative_results"])),
  author_limitations: asText(firstValue(record, ["author_limitations", "limitations"])),
  evidence_locations: asText(firstValue(record, ["evidence_locations", "evidence"])),
  access_attempts: asText(firstValue(record, ["access_attempts", "access_log"])),
  reading_method: asText(firstValue(record, ["reading_method", "reading_medium", "read_by"])),
  notes: asText(firstValue(record, ["notes", "note"])),
}));

const statusFill = {
  FULL_TEXT: ["#E2F0D9", "#2F6B2F"],
  ABSTRACT_ONLY: ["#FFF2CC", "#7F6000"],
  METADATA_ONLY: ["#FCE4D6", "#9E480E"],
  PARTIAL_TEXT: ["#E4DFEC", "#5B2C6F"],
};

const styleSheet = (sheet, usedAddress, headerAddress, bodyAddress, widths, rowHeight) => {
  sheet.showGridLines = false;
  const used = sheet.getRange(usedAddress);
  used.format.font = { name: FONT, size: 10, color: "#1F2937" };
  used.format.wrapText = true;
  used.format.verticalAlignment = "top";
  used.format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  const header = sheet.getRange(headerAddress);
  header.format = {
    fill: "#1F4E78",
    font: { name: FONT, size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  if (bodyAddress) {
    const body = sheet.getRange(bodyAddress);
    body.format.verticalAlignment = "top";
    body.format.wrapText = true;
    body.format.rowHeight = rowHeight;
  }
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}1:${column}100`).format.columnWidth = width;
  }
  sheet.getRange(headerAddress).format.rowHeight = 38;
};

const displayWidth = (value) => [...asText(value)].reduce((width, character) => {
  const codePoint = character.codePointAt(0) ?? 0;
  return width + (codePoint <= 0x7F ? 1 : 2);
}, 0);

const estimateWrappedLines = (value, width) => {
  const text = asText(value);
  if (!text) return 1;
  const effectiveCapacity = Math.max(8, width * 0.95);
  return text.split(/\r?\n/).reduce((count, line) => {
    return count + Math.max(1, Math.ceil(displayWidth(line) / effectiveCapacity));
  }, 0);
};

const fitWrappedRows = (sheet, startRow, rows, widths, minimumHeight) => {
  const columns = Object.keys(widths);
  const lastColumn = columns[columns.length - 1];
  rows.forEach((row, index) => {
    const lineCount = row.reduce((maxLines, value, columnIndex) => {
      const column = columns[columnIndex];
      return Math.max(maxLines, estimateWrappedLines(value, widths[column] ?? 12));
    }, 1);
    const height = Math.min(409, Math.max(minimumHeight, Math.ceil(10 + lineCount * 14)));
    const rowNumber = startRow + index;
    sheet.getRange(`A${rowNumber}:${lastColumn}${rowNumber}`).format.rowHeight = height;
  });
};

const colorStatus = (sheet, column, startRow, rows) => {
  rows.forEach((row, index) => {
    const status = asText(row.reading_status);
    const [fill, color] = statusFill[status] ?? ["#F2F2F2", "#595959"];
    const cell = sheet.getRange(`${column}${startRow + index}`);
    cell.format.fill = fill;
    cell.format.font = { name: FONT, size: 10, bold: true, color };
  });
};

const inspectAndSave = async (workbook, spec, fileName) => {
  const result = await workbook.inspect(spec);
  await fs.writeFile(path.join(QA_DIR, fileName), result.ndjson ?? JSON.stringify(result, null, 2), "utf8");
  return result;
};

const main = async () => {
  await fs.mkdir(QA_DIR, { recursive: true });
  const [a, bRaw, c, critique] = await Promise.all([
    readJson("track_A.json"),
    readJson("track_B.json"),
    readJson("track_C.json"),
    readJson("primary_critique.json"),
  ]);
  const b = Array.isArray(bRaw) ? bRaw : (bRaw.documents ?? []);
  const rawRecords = [...a, ...b, ...c];
  const records = makeWorkbookRelativeData(rawRecords).sort((left, right) => left.id.localeCompare(right.id, undefined, { numeric: true }));

  if (records.length !== 35) throw new Error(`Expected 35 records, found ${records.length}`);
  const ids = records.map((record) => record.id);
  if (new Set(ids).size !== ids.length) throw new Error("Duplicate literature IDs found");
  const markerCount = records.reduce((count, record) => count + ((record.authors.match(/★/g) ?? []).length), 0);
  if (markerCount !== records.length) throw new Error(`First-author marker count ${markerCount} != ${records.length}`);
  const fullTextRecords = records.filter((record) => record.reading_status === "FULL_TEXT");
  const abstractOnlyRecords = records.filter((record) => record.reading_status === "ABSTRACT_ONLY");
  if (fullTextRecords.length !== 33) throw new Error(`Expected 33 FULL_TEXT records, found ${fullTextRecords.length}`);
  if (abstractOnlyRecords.length !== 2) throw new Error(`Expected 2 ABSTRACT_ONLY records, found ${abstractOnlyRecords.length}`);
  if (abstractOnlyRecords.map((record) => record.id).sort().join(",") !== "A06,C03") {
    throw new Error(`Unexpected ABSTRACT_ONLY IDs: ${abstractOnlyRecords.map((record) => record.id).join(",")}`);
  }
  const localPdfRecords = records.filter((record) => record.local_pdf_relative);
  if (localPdfRecords.length !== 31) throw new Error(`Expected 31 local main PDF paths, found ${localPdfRecords.length}`);

  const reviews = records.map((record) => {
    const current = critiqueText(critique[record.id]);
    return [
      record.id,
      titleWithYear(record),
      joinParts(addLabel("方法", record.method), addLabel("场景/设定", record.scenario)),
      joinParts(addLabel("主要结论", record.findings), addLabel("定量结果", record.quantitative_findings)),
      record.author_limitations,
      current.assessment,
      current.relevance,
      record.reading_status,
    ];
  });

  const bibliography = records.map((record) => [
    record.id,
    record.authors,
    record.venue,
    record.year,
    record.doi,
    record.primary_url,
    record.fulltext_url,
    record.local_pdf_relative,
    record.local_text_relative,
    record.version,
    record.scenario,
    record.quantitative_findings,
    record.evidence_locations,
    record.reading_status,
    asText(record.publication_status),
    asText(record.metadata_status),
    joinParts(addLabel("历史访问尝试", record.access_attempts), addLabel("备注", record.notes)),
  ]);

  const notFull = records.filter((record) => record.reading_status !== "FULL_TEXT");
  const remoteReadWithoutLocal = records.filter((record) => record.reading_status === "FULL_TEXT" && !record.local_pdf_relative);
  const missingRows = [
    ...notFull.map((record) => [
      record.id,
      titleWithYear(record),
      record.reading_status,
      record.local_pdf_relative,
      record.local_text_relative,
      record.primary_url,
      record.fulltext_url,
      joinParts(addLabel("访问记录", record.access_attempts), addLabel("备注", record.notes)),
      "未阅读全文；按用户要求停止追索，不作为研究推进前提",
    ]),
    ...remoteReadWithoutLocal.map((record) => [
      record.id,
      titleWithYear(record),
      record.reading_status,
      record.local_pdf_relative,
      record.local_text_relative,
      record.primary_url,
      record.fulltext_url,
      joinParts(addLabel("阅读媒介", record.reading_method), addLabel("访问记录", record.access_attempts), addLabel("备注", record.notes)),
      `已全文阅读（${record.reading_method || "远程全文"}），但本地 PDF/文本文件缺失；此行不计入未全文阅读数量`,
    ]),
  ];

  const workbook = Workbook.create();
  const reviewSheet = workbook.worksheets.add("文献梳理");
  const bibSheet = workbook.worksheets.add("书目与证据");
  const missingSheet = workbook.worksheets.add("全文获取状态");

  const reviewHeaders = ["ID", "题名（年）", "做了什么", "主要结论", "作者讨论／原文设定边界", "本次评述：不足与待核实点", "研究启示", "阅读状态"];
  reviewSheet.getRange(`A1:H${reviews.length + 1}`).values = [reviewHeaders, ...reviews];
  const reviewTable = reviewSheet.tables.add(`A1:H${reviews.length + 1}`, true, "LiteratureReviewTable");
  reviewTable.showFilterButton = true;
  reviewSheet.freezePanes.freezeRows(1);
  reviewSheet.freezePanes.freezeColumns(1);
  const reviewWidths = { A: 9, B: 34, C: 62, D: 62, E: 52, F: 52, G: 45, H: 18 };
  styleSheet(reviewSheet, `A1:H${reviews.length + 1}`, "A1:H1", `A2:H${reviews.length + 1}`, reviewWidths, 150);
  fitWrappedRows(reviewSheet, 2, reviews, reviewWidths, 72);
  colorStatus(reviewSheet, "H", 2, records);
  reviewSheet.getRange("J1:K3").values = [
    ["表说明", "实验边界不等于作者自评；本次评价独立列示；ABSTRACT_ONLY按用户要求停止追索，仅按摘要证据使用。"],
    ["作者讨论／原文设定边界", "保留 evidence JSON 中的原字段条目措辞，不将实验边界统一归因于作者自评。"],
    ["阅读状态", "FULL_TEXT、ABSTRACT_ONLY、METADATA_ONLY、PARTIAL_TEXT 等状态沿用证据记录。"],
  ];
  reviewSheet.getRange("J1:J3").format = { fill: "#D9EAF7", font: { name: FONT, size: 10, bold: true, color: "#1F2937" }, wrapText: true, verticalAlignment: "top" };
  reviewSheet.getRange("K1:K3").format = { font: { name: FONT, size: 10, color: "#1F2937" }, wrapText: true, verticalAlignment: "top" };
  reviewSheet.getRange("J1:K3").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  reviewSheet.getRange("J1:J3").format.columnWidth = 28;
  reviewSheet.getRange("K1:K3").format.columnWidth = 74;
  reviewSheet.getRange("J1:K3").format.rowHeight = 52;
  reviewSheet.tabColor = "#1F4E78";

  const bibHeaders = ["ID", "完整作者（★为第一作者）", "载体", "年份", "DOI", "主来源URL", "全文/下载URL", "本地PDF路径（相对本表）", "本地文本路径（相对本表）", "版本", "场景/规模/假设", "定量结果", "证据定位", "阅读状态", "出版状态", "元数据状态", "访问与备注"];
  bibSheet.getRange(`A1:Q${bibliography.length + 1}`).values = [bibHeaders, ...bibliography];
  const bibliographyTable = bibSheet.tables.add(`A1:Q${bibliography.length + 1}`, true, "BibliographyEvidenceTable");
  bibliographyTable.showFilterButton = true;
  bibSheet.freezePanes.freezeRows(1);
  bibSheet.freezePanes.freezeColumns(1);
  const bibliographyWidths = { A: 9, B: 42, C: 32, D: 10, E: 30, F: 48, G: 52, H: 48, I: 48, J: 14, K: 58, L: 52, M: 42, N: 18, O: 18, P: 18, Q: 58 };
  styleSheet(bibSheet, `A1:Q${bibliography.length + 1}`, "A1:Q1", `A2:Q${bibliography.length + 1}`, bibliographyWidths, 125);
  fitWrappedRows(bibSheet, 2, bibliography, bibliographyWidths, 72);
  colorStatus(bibSheet, "N", 2, records);
  bibSheet.tabColor = "#548235";

  missingSheet.mergeCells("A1:I1");
  missingSheet.mergeCells("A2:I2");
  missingSheet.mergeCells("A3:I3");
  missingSheet.getRange("A1").values = [[`未全文阅读：${notFull.length}篇`]];
  const remoteReadSummary = remoteReadWithoutLocal
    .map((record) => `${record.id}（${record.reading_method || "远程全文"}）`)
    .join("、");
  missingSheet.getRange("A2").values = [[`已读全文但无本地文件：${remoteReadWithoutLocal.length}篇`]];
  missingSheet.getRange("A3").values = [[`说明：${remoteReadSummary} 已完成全文阅读，但因本地离线文件缺失而单独列示；不计入未全文阅读数量。`]];
  missingSheet.getRange("A1:I3").format.font = { name: FONT, size: 10, color: "#1F2937" };
  missingSheet.getRange("A1:I2").format.font = { name: FONT, size: 10, bold: true, color: "#1F2937" };
  missingSheet.getRange("A1:I3").format.wrapText = true;
  missingSheet.getRange("A1:I3").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  missingSheet.getRange("A1:I3").format.fill = "#D9EAF7";
  missingSheet.getRange("A1:I3").format.verticalAlignment = "top";
  missingSheet.getRange("A1:I2").format.rowHeight = 30;
  missingSheet.getRange("A3:I3").format.rowHeight = 58;
  const missingHeaders = ["ID", "题名（年）", "阅读状态", "本地PDF路径（相对本表）", "本地文本路径（相对本表）", "主来源URL", "全文/下载URL", "历史访问尝试与备注", "处理说明"];
  const tableStart = 5;
  const tableEnd = tableStart + missingRows.length;
  missingSheet.getRange(`A${tableStart}:I${tableEnd}`).values = [missingHeaders, ...missingRows];
  const missingTable = missingSheet.tables.add(`A${tableStart}:I${tableEnd}`, true, "MissingFullTextTable");
  missingTable.showFilterButton = true;
  missingSheet.freezePanes.freezeRows(tableStart);
  missingSheet.freezePanes.freezeColumns(1);
  const missingWidths = { A: 9, B: 38, C: 18, D: 48, E: 48, F: 48, G: 52, H: 60, I: 58 };
  styleSheet(missingSheet, `A${tableStart}:I${tableEnd}`, `A${tableStart}:I${tableStart}`, `A${tableStart + 1}:I${tableEnd}`, missingWidths, 105);
  fitWrappedRows(missingSheet, tableStart + 1, missingRows, missingWidths, 72);
  const missingStatusRows = missingRows.map((row) => ({ reading_status: row[2] }));
  colorStatus(missingSheet, "C", tableStart + 1, missingStatusRows);
  const remoteOnlyStart = tableStart + 1 + notFull.length;
  if (remoteReadWithoutLocal.length > 0) {
    const remoteOnlyEnd = remoteOnlyStart + remoteReadWithoutLocal.length - 1;
    missingSheet.getRange(`C${remoteOnlyStart}:I${remoteOnlyEnd}`).format.fill = "#DDEBF7";
  }
  missingSheet.tabColor = "#C55A11";

  workbook.recalculate();

  await inspectAndSave(workbook, {
    kind: "table",
    range: `文献梳理!A1:H${reviews.length + 1}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 8,
    tableMaxCellChars: 160,
  }, "inspect_review.ndjson");
  await inspectAndSave(workbook, {
    kind: "table",
    range: `书目与证据!A1:Q${bibliography.length + 1}`,
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 17,
    tableMaxCellChars: 160,
  }, "inspect_bibliography.ndjson");
  await inspectAndSave(workbook, {
    kind: "table",
    range: `全文获取状态!A1:I${tableEnd}`,
    include: "values,formulas",
    tableMaxRows: Math.min(20, missingRows.length + 5),
    tableMaxCols: 9,
    tableMaxCellChars: 160,
  }, "inspect_missing.ndjson");
  await inspectAndSave(workbook, {
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  }, "formula_error_scan.ndjson");

  const renderSpecs = [
    ["文献梳理", "A1:K8", "review_preview.png"],
    ["书目与证据", "A1:Q7", "bibliography_preview.png"],
    ["全文获取状态", `A1:I${Math.min(tableEnd, 12)}`, "missing_preview.png"],
  ];

  const updatedIds = new Set(["A03", "A11", "A12", "B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "C02", "C04", "C06"]);
  const updatedRows = records
    .map((record, index) => ({ id: record.id, row: index + 2 }))
    .filter(({ id }) => updatedIds.has(id));
  const groups = [];
  for (const item of updatedRows) {
    const previous = groups[groups.length - 1];
    if (previous && item.row === previous.end + 1) previous.end = item.row;
    else groups.push({ start: item.row, end: item.row });
  }
  renderSpecs.push(["文献梳理", "A1:D8", "review_top_left.png"]);
  renderSpecs.push(["文献梳理", "E1:H8", "review_top_right.png"]);
  for (const group of groups) {
    renderSpecs.push(["文献梳理", `A${group.start}:D${group.end}`, `review_updated_${group.start}_${group.end}_left.png`]);
    renderSpecs.push(["文献梳理", `E${group.start}:H${group.end}`, `review_updated_${group.start}_${group.end}_right.png`]);
  }
  for (const [sheetName, range, fileName] of renderSpecs) {
    const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
    await fs.writeFile(path.join(QA_DIR, fileName), new Uint8Array(await preview.arrayBuffer()));
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT);

  const manifest = {
    record_count: records.length,
    full_text_count: fullTextRecords.length,
    abstract_only_count: abstractOnlyRecords.length,
    abstract_only_ids: abstractOnlyRecords.map((record) => record.id),
    local_main_pdf_count: localPdfRecords.length,
    not_full_count: notFull.length,
    remote_read_without_local_file_count: remoteReadWithoutLocal.length,
    missing_sheet_row_count: missingRows.length,
    ids,
    first_author_marker_count: markerCount,
    workbook: OUTPUT,
  };
  await fs.writeFile(path.join(QA_DIR, "build_manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  console.log(JSON.stringify(manifest, null, 2));
};

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exitCode = 1;
});
