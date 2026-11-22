"""Acquire a separate, pinned multi-company corpus for trajectory experiments."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import argparse
import json
from pathlib import Path
import re

from .filings import CHINA, DATA, MANIFEST, REPORTS, download, read_url

OUTPUT = DATA / "trajectories" / "v1"
# Issuer splits are assigned before question construction. The phase-one corpus stays intact.
ISSUERS = {
    "000333": ("美的集团", "train", "家电"),
    "000651": ("格力电器", "train", "家电"),
    "300750": ("宁德时代", "train", "动力电池"),
    "000063": ("中兴通讯", "train", "通信设备"),
    "002594": ("比亚迪", "validation", "汽车"),
    "600900": ("长江电力", "validation", "电力"),
    "600690": ("海尔智家", "train", "家电"),
    "600519": ("贵州茅台", "test", "白酒"),
    "600887": ("伊利股份", "test", "乳制品"),
}


def acquire(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "sources.json"
    reports = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else []
    expected_ids = {f"{code}-{year}-{period}" for code in ISSUERS for year in (2023, 2024, 2025) for period in ("FY", "H1")}
    if {r["id"] for r in reports} != expected_ids or any(r["split"] != ISSUERS[r["company_code"]][1] for r in reports):
        stocks = json.loads(read_url("https://www.cninfo.com.cn/new/data/szse_stock.json"))["stockList"]
        orgs = {r["code"]: r["orgId"] for r in stocks}
        original = {r["id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
        reports = []
        for code, (name, split, industry) in ISSUERS.items():
            form = dict(pageNum=1, pageSize=100, column="sse" if code.startswith("6") else "szse",
                        tabName="fulltext", plate="", stock=f"{code},{orgs[code]}", searchkey="", secid="",
                        category="category_ndbg_szsh;category_bndbg_szsh;", trade="",
                        seDate="2023-07-01~2026-07-01", sortName="time", sortType="desc", isHLtitle="false")
            response = json.loads(read_url("https://www.cninfo.com.cn/new/hisAnnouncement/query", form))
            found = {}
            for row in response.get("announcements") or []:
                title = re.sub(r"\s+|<[^>]+>", "", row["announcementTitle"])
                match = re.search(r"(202[345])年(年度|半年度)报告(?:全文)?$", title)
                if row["secCode"] != code or not match:
                    continue
                year, period = int(match[1]), "H1" if match[2] == "半年度" else "FY"
                report_id = f"{code}-{year}-{period}"
                if report_id in found:
                    continue
                relative = row["adjunctUrl"]
                if not relative.startswith("finalpage/") or not relative.lower().endswith(".pdf"):
                    continue
                found[report_id] = original.get(report_id) or {
                    "id": report_id, "company_code": code, "company": name, "year": year, "period": period,
                    "published_at": datetime.fromtimestamp(row["announcementTime"] / 1000, CHINA).date().isoformat(),
                    "title": title, "url": "https://static.cninfo.com.cn/" + relative,
                    "announcement_id": row["announcementId"], "version": "公开全文",
                }
            if len(found) != 6:
                raise ValueError(f"{name}: expected six reports, got {sorted(found)}")
            reports.extend(dict(r, split=split, industry=industry) for r in found.values())
            print(name, len(found), flush=True)
        reports.sort(key=lambda r: r["id"])
        # Save discovery before downloading so interrupted acquisitions can resume.
        manifest_file.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        acquired = list(pool.map(download, reports))
    manifest_file.write_text(json.dumps(acquired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(acquired)} PDFs", flush=True)
    return acquired


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    acquire(parser.parse_args().output)
