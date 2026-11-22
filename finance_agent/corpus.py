"""Build and query the SQLite evidence index."""
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
import re
import sqlite3

import jieba
import pypdfium2 as pdf

from .filings import MANIFEST, REPORTS, RUNTIME

DB = RUNTIME / "filings.sqlite3"
SUMMARY = "主要会计数据和财务指标"
SEGMENTS = "营业收入构成 · 分产品"
METRICS = {
    "revenue": "营业收入",
    "profit": "归属于上市公司股东的净利润",
    "cash": "经营活动产生的现金流量净额",
}
REPORT_KIND = {"FY": "年报 年度报告 全年", "H1": "半年报 半年度报告 上半年"}
MONEY = r"-?\d[\d,]*(?:\.\d+)?"
SUMMARY_PAGES = {
    "000333": (9, 7, 9, 7), "000651": (7, 7, 7, 7), "600690": (10, 6, 12, 6),
}
SEGMENT_PAGES = {
    "000333-2024-FY": 49, "000333-2024-H1": 46,
    "000333-2025-FY": 49, "000333-2025-H1": 55,
    "000651-2024-FY": 25, "000651-2024-H1": 21,
    "000651-2025-FY": 24, "000651-2025-H1": 22,
    "600690-2024-FY": 42, "600690-2025-FY": 42,
}
PRODUCTS = {
    "000333": ("智能家居业务", "商业及工业解决方案", "其他"),
    "000333-2024-H1": ("暖通空调", "消费电器", "机器人、自动化系统及其他制造业"),
    "000651": ("消费电器", "工业制品及绿色能源", "智能装备", "其他主营", "其他业务"),
    "000651-2024-H1": ("空调", "生活电器", "工业制品", "智能装备", "绿色能源", "其他主营", "其他业务"),
    "600690": ("空调", "电冰箱", "厨电", "水家电", "洗衣机", "装备部品及渠道综合服务"),
}


def tokens(text):
    return [word.lower() for word in jieba.cut_for_search(text) if re.fullmatch(r"[\w\u4e00-\u9fff]+", word)]


def amount(value):
    return str(Decimal(value.replace(",", "")))


def get_pages(path):
    with pdf.PdfDocument(path) as document:
        for page in document:
            textpage = page.get_textpage()
            try:
                yield textpage.get_text_bounded()
            finally:
                textpage.close()
                page.close()


def table_row(text, label, pattern):
    # Insert whitespace tolerance only between label characters. Keep amount
    # boundaries intact so adjacent financial columns cannot merge.
    name = r"\s*".join(map(re.escape, label))
    return re.search(rf"(?:^|\n)\s*{name}\s*(?:[（(](?:千元|元)[）)])?\s+{pattern}", text)


def structured(report, pages):
    code, report_id = report["company_code"], report["id"]
    unit = "千元" if code == "000333" else "元"
    restated = code == "600690"
    page = SUMMARY_PAGES[code][(report["year"] - 2024) * 2 + (report["period"] == "H1")]
    for key, label in METRICS.items():
        skip = rf"\s+{MONEY}" if restated else ""
        pattern = rf"({MONEY})\s+({MONEY}){skip}\s+({MONEY})%?"
        row = table_row(pages[page - 1], label, pattern)
        if row is None:
            raise ValueError(f"{report_id} PDF {page}: could not verify {label}")
        current, prior, yoy = map(amount, row.groups())
        note = "同比采用本报告的上年同期调整后列" if restated else "同比采用本报告的上年同期列"
        yield page, SUMMARY, f"{label} 本期 {current} {unit} 上年同期 {prior} {unit} 同比 {yoy}% · {note}", {
            "kind": "metric", "metric": key, "label": label, "current": current, "prior": prior,
            "yoy": yoy, "unit": unit, "scope": "合并", "note": note,
        }
    if report_id not in SEGMENT_PAGES:
        return
    page = SEGMENT_PAGES[report_id]
    text = pages[page - 1]
    # Prefer the full product mix table to the separate >10% table.
    if code == "600690":
        text = text.split("主营业务分产品情况", 1)[-1].split("主营业务分地区情况", 1)[0]
    else:
        if report_id != "000333-2025-H1":
            text = text.split("分产品", 1)[-1]
        text = text.split("分地区", 1)[0]
    rows = []
    for label in PRODUCTS.get(report_id, PRODUCTS[code]):
        if code == "600690":
            row = table_row(text, label, rf"({MONEY})\s+{MONEY}\s+{MONEY}\s+({MONEY})")
            if row is None:
                raise ValueError(f"{report_id}: missing product {label}")
            current, yoy = map(amount, row.groups())
            prior = None
        else:
            row = table_row(text, label, rf"({MONEY})\s+{MONEY}%\s+({MONEY})\s+{MONEY}%\s+({MONEY})%")
            if row is None:
                raise ValueError(f"{report_id}: missing product {label}")
            current, prior, yoy = map(amount, row.groups())
        rows.append({"label": label, "current": current, "prior": prior, "yoy": yoy, "unit": unit})
    rows.sort(key=lambda row: Decimal(row["current"]), reverse=True)
    content = "；".join(
        f"{row['label']} {row['current']} {unit} 同比 {row['yoy']}%" for row in rows
    )
    yield page, SEGMENTS, content, {"kind": "segments", "products": rows}


def build():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    target = DB.with_suffix(".building")
    target.unlink(missing_ok=True)
    reports = json.loads(MANIFEST.read_text(encoding="utf-8"))
    con = sqlite3.connect(target)
    try:
        with con:
            con.executescript("""
                CREATE TABLE evidence (id TEXT PRIMARY KEY, report_id TEXT, company TEXT,
                    company_code TEXT, year INTEGER, period TEXT, published_at TEXT, url TEXT,
                    page INTEGER, section TEXT, contents TEXT, fact TEXT);
                CREATE VIRTUAL TABLE search_index USING fts5(evidence_id UNINDEXED, terms);
            """)
            for report in reports:
                pages = list(get_pages(REPORTS / f"{report['id']}.pdf"))
                rows = list(structured(report, pages))
                for page, body in enumerate(pages, 1):
                    body = " ".join(body.split())
                    for offset in range(0, len(body), 900):
                        rows.append((page, "原文", body[offset:offset + 1100], None))
                for i, (page, section, body, fact) in enumerate(rows):
                    evidence_id = f"{report['id']}:{page}:{i}"
                    data = [report[field] for field in ("id", "company", "company_code", "year", "period", "published_at", "url")]
                    con.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                        evidence_id, *data, page, section, body,
                        json.dumps(fact, ensure_ascii=False) if fact else None,
                    ))
                    header = f"{report['company']} {report['year']} {REPORT_KIND[report['period']]} {section}"
                    con.execute("INSERT INTO search_index VALUES (?,?)", (evidence_id, " ".join(tokens(f"{header} {body}"))))
                print(report["id"], len(pages), len(rows), flush=True)
    finally:
        con.close()
    target.replace(DB)


@dataclass(frozen=True)
class Scope:
    company_code: str | None = None
    as_of: str | None = None
    year: int | None = None
    period: str | None = None


def retrieve(query, topk=3, scope=Scope()):
    words = list(dict.fromkeys(tokens(query)))[:45]
    if not words:
        return []
    expression = " OR ".join(f'"{word}"' for word in words)
    sql = """
        SELECT e.*, -bm25(search_index) * (CASE WHEN e.fact IS NULL THEN 1 ELSE 3 END) AS score
        FROM search_index JOIN evidence e ON e.id = search_index.evidence_id
        WHERE search_index MATCH ?
    """
    values = [expression]
    for field, value in vars(scope).items():
        if value is not None:
            sql += " AND " + ("e.published_at <= ?" if field == "as_of" else f"e.{field} = ?")
            values.append(value)
    sql += " ORDER BY score DESC, e.id LIMIT ?"
    con = sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        results = [dict(row) for row in con.execute(sql, [*values, topk])]
    finally:
        con.close()
    for row in results:
        row["fact"] = json.loads(row["fact"]) if row["fact"] else None
    return results


if __name__ == "__main__":
    build()
