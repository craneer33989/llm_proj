"""Extract auditable summary rows and build an independent FTS corpus."""
from decimal import Decimal, ROUND_HALF_UP
import argparse
import json
from pathlib import Path
import re
import sqlite3

from .corpus import DB as LEGACY_DB, get_pages, tokens
from .filings import REPORTS
from .prepare_sft import sha256, write_jsonl
from .trajectory_sources import OUTPUT

LABELS = {
    "revenue": ("营业收入",),
    "profit": ("归属于上市公司股东的净利润", "归属于上市公司普通股股东的净利润", "归属于母公司所有者的净利润", "归属于上市公司股东的净利潤"),
    "cash": ("经营活动产生的现金流量净额", "经营活动现金流量净额"),
}
NAMES = {"revenue": "营业收入", "profit": "归母净利润", "cash": "经营活动产生的现金流量净额"}
MULTIPLIERS = {"元": 1, "千元": 1000, "万元": 10000, "百万元": 1000000, "亿元": 100000000}
NUMBER = r"[−\-]?\d[\d,]*(?:\.\d+)?"
UNIT = r"百万元|亿元|万元|千元|元"
SPACED_UNIT = "|".join(r"\s*".join(word) for word in MULTIPLIERS)


def decimal(text):
    return Decimal(text.replace(",", "").replace("−", "-"))


def extract_metric(report, pages, metric):
    for page_no, page in enumerate(pages[:25], 1):
        for label in LABELS[metric]:
            spaced = r"\s*".join(map(re.escape, label))
            pattern = (rf"(?:^|\n)\s*{spaced}\s*(?:[（(]\s*(?:人民币)?(?P<unit>{SPACED_UNIT})\s*[）)])?"
                       rf"\s+(?P<run>{NUMBER}(?:\s+[（(]?{NUMBER}(?:\s*%)?[）)]?){{1,5}})")
            match = re.search(pattern, page)
            if not match:
                continue
            context = (pages[page_no - 2] if page_no > 1 else "") + page[:match.start()]
            compact_context = re.sub(r"\s+", "", context)
            if not any(label in compact_context for label in ("主要会计数据", "主要财务数据")):
                continue
            unit = re.sub(r"\s+", "", match["unit"]) if match["unit"] else None
            if not unit:
                units = list(re.finditer(rf"单位\s*[:：]\s*(?:人民币)?\s*({UNIT})", context))
                if not units:
                    continue  # Never guess the unit from the company's identity.
                unit = units[-1][1]
            cells = list(re.finditer(rf"(?P<negative>[（(])?(?P<value>{NUMBER})\s*(?P<percent>%)?[）)]?", match["run"]))
            not_applicable = len(cells) == 2 and "不适用" in page[match.end():match.end() + 12]
            percent_index = next((i for i, cell in enumerate(cells) if cell["percent"]), None)
            if percent_index is None:
                # SSE tables state (%) in the heading, not in each percentage cell.
                if "增减" not in compact_context or "%" not in compact_context:
                    continue
                pairs = list(re.finditer(r"调整(前|后)\s+调整(前|后)", context))
                percent_index = 3 if pairs else 2
            if percent_index not in (2, 3) or (len(cells) <= percent_index and not not_applicable):
                continue
            prior_index = 1
            if percent_index == 3:
                pairs = list(re.finditer(r"调整(前|后)\s+调整(前|后)", context))
                if not pairs:
                    continue
                prior_index = 1 if pairs[0][1] == "后" else 2
            current, prior = decimal(cells[0]["value"]), decimal(cells[prior_index]["value"])
            yoy_cell = cells[percent_index] if not not_applicable else None
            yoy = decimal(yoy_cell["value"]) * (-1 if yoy_cell["negative"] else 1) if yoy_cell else None
            computed = ((current - prior) / prior * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if prior > 0 else None
            if computed is not None and yoy is not None and abs(computed - yoy) > Decimal("0.03"):
                raise ValueError(f"{report['id']} {metric}: disclosed/computed growth differ: {yoy}/{computed}")
            fact = {"kind": "metric", "metric": metric, "label": NAMES[metric],
                    "current": str(current * MULTIPLIERS[unit]), "prior": str(prior * MULTIPLIERS[unit]),
                    "yoy": str(yoy) if yoy is not None else None, "unit": "元", "original_unit": unit,
                    "restated": percent_index == 3,
                    "note": "同期值采用本报告可比列（存在调整时采用调整后列）；跨报告原披露值变化不等于披露同比。"}
            return dict(report, id=f"{report['id']}:{page_no}:{metric}", report_id=report["id"],
                        page=page_no, section="主要会计数据和财务指标", fact=fact,
                        contents=f"{NAMES[metric]} 本期 {fact['current']} 元 同期可比 {fact['prior']} 元 披露同比 {str(yoy) + '%' if yoy is not None else '不适用'}",
                        raw_excerpt=match[0].strip(), extraction_check={"computed_yoy": str(computed) if computed is not None else None,
                            "disclosed_yoy": str(yoy) if yoy is not None else None, "prior_column": prior_index})
    raise ValueError(f"{report['id']}: no unambiguous summary row for {metric}")


def build(output=OUTPUT):
    reports = json.loads((output / "sources.json").read_text(encoding="utf-8"))
    target = output / "corpus.sqlite3"
    staging = output / "corpus.building.sqlite3"
    staging.unlink(missing_ok=True)
    catalog, summary = [], []
    con = sqlite3.connect(staging)
    legacy = sqlite3.connect(LEGACY_DB.as_uri() + "?mode=ro", uri=True)
    legacy.row_factory = sqlite3.Row
    try:
        con.executescript("""CREATE TABLE evidence (id TEXT PRIMARY KEY, report_id TEXT, company_code TEXT,
            published_at TEXT, year INTEGER, period TEXT, data TEXT);
            CREATE VIRTUAL TABLE search_index USING fts5(id UNINDEXED, terms);""")
        for report in reports:
            path = REPORTS / (report["id"] + ".pdf")
            if sha256(path) != report["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            pages = list(get_pages(path))
            facts = [extract_metric(report, pages, metric) for metric in LABELS]
            # Cross-check all existing summary facts against the independently built phase-one index.
            for row in facts:
                prior = legacy.execute("SELECT fact FROM evidence WHERE report_id=? AND fact IS NOT NULL", (report["id"],)).fetchall()
                for item in prior:
                    fact = json.loads(item[0])
                    if fact.get("metric") == row["fact"]["metric"]:
                        for field in ("current", "prior", "yoy"):
                            old = Decimal(fact[field]) * (MULTIPLIERS[fact["unit"]] if field != "yoy" else 1)
                            if old != Decimal(row["fact"][field]):
                                raise ValueError(f"Legacy cross-check failed: {row['id']} / {field}")
            # Preserve product tables with their original verified labels and page citations.
            for item in legacy.execute("SELECT * FROM evidence WHERE report_id=? AND fact IS NOT NULL", (report["id"],)):
                fact = json.loads(item["fact"])
                if fact["kind"] != "segments":
                    continue
                for product in fact["products"]:
                    multiplier = MULTIPLIERS[product["unit"]]
                    product["current"] = str(Decimal(product["current"]) * multiplier)
                    product["prior"] = str(Decimal(product["prior"]) * multiplier) if product["prior"] is not None else None
                    product["unit"] = "元"
                facts.append(dict(report, id=f"{report['id']}:{item['page']}:segments", report_id=report["id"],
                                  page=item["page"], section=item["section"], contents=item["contents"], fact=fact,
                                  raw_excerpt=pages[item["page"] - 1]))
            catalog.extend(facts)
            all_rows = list(facts)
            for page_no, text in enumerate(pages, 1):
                text = " ".join(text.split())
                for offset in range(0, len(text), 900):
                    all_rows.append(dict(report, id=f"{report['id']}:{page_no}:text-{offset}", report_id=report["id"],
                                         page=page_no, section="原文", contents=text[offset:offset + 1100], fact=None))
            for row in all_rows:
                con.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?)", (
                    row["id"], row["report_id"], row["company_code"], row["published_at"], row["year"], row["period"],
                    json.dumps(row, ensure_ascii=False)))
                header = f"{row['company']} {row['year']} {row['period']} {row['section']}"
                con.execute("INSERT INTO search_index VALUES (?,?)", (row["id"], " ".join(tokens(header + " " + row["contents"]))))
            summary.append({"report_id": report["id"], "pages": len(pages), "records": len(all_rows), "facts": len(facts)})
            print(report["id"], len(pages), [f["fact"].get("metric", "segments") for f in facts], flush=True)
        con.commit()
    finally:
        con.close()
        legacy.close()
    staging.replace(target)
    write_jsonl(output / "fact-catalog.jsonl", catalog)
    (output / "corpus-manifest.json").write_text(json.dumps({"reports": summary, "pdf_count": len(reports),
        "pages": sum(r["pages"] for r in summary), "records": sum(r["records"] for r in summary),
        "facts": len(catalog), "sources_sha256": sha256(output / "sources.json"), "corpus_sha256": sha256(target)},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def search(query, *, company_code, year, period, as_of, output=OUTPUT, topk=5):
    if not isinstance(query, str) or not query.strip() or len(query) > 240:
        raise ValueError("Search requires 1..240 query characters")
    if not isinstance(company_code, str) or type(year) is not int or period not in ("FY", "H1"):
        raise ValueError("Search requires company_code, integer year, and FY/H1 period")
    # Interleave keyword groups so a long cash-flow label cannot crowd a short revenue label out.
    groups = list(dict.fromkeys(query.split()))[:8]
    con = sqlite3.connect((output / "corpus.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        rankings = []
        for group in groups:
            words = list(dict.fromkeys(tokens(group)))[:45]
            expression = " OR ".join('"' + word + '"' for word in words)
            if not expression:
                continue
            rows = con.execute("""SELECT e.data FROM search_index s JOIN evidence e ON e.id=s.id
                WHERE search_index MATCH ? AND e.company_code=? AND e.year=? AND e.period=? AND e.published_at<=?
                ORDER BY bm25(search_index) * CASE WHEN json_extract(e.data,'$.fact') IS NULL THEN 1 ELSE 3 END, e.id LIMIT ?""",
                (expression, company_code, year, period, as_of, topk * 2)).fetchall()
            rankings.append([json.loads(item[0]) for item in rows])
        found = {}
        for rank in range(topk * 2):
            for ranking in rankings:
                if rank < len(ranking):
                    row = ranking[rank]
                    found.setdefault(row["id"], row)
                    if len(found) == topk:
                        return list(found.values())
        return list(found.values())
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    build(parser.parse_args().output)
