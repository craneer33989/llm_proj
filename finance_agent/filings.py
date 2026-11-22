"""Acquire official filings and build a reproducible local corpus."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = DATA / "reports"
RUNTIME = DATA / "runtime"
MANIFEST = DATA / "filings.json"
COMPANIES = {
    "000333": ("美的集团", "szse", "9900005965"),
    "000651": ("格力电器", "szse", "gssz0000651"),
    "600690": ("海尔智家", "sse", "gssh0600690"),
}
YEARS = (2024, 2025)
CNINFO = "https://www.cninfo.com.cn"
STATIC = "https://static.cninfo.com.cn"
CHINA = timezone(timedelta(hours=8))


def read_url(url, form=None):
    request = Request(url, data=urlencode(form).encode() if form else None, headers={
        "User-Agent": "Mozilla/5.0", "Referer": f"{CNINFO}/",
    })
    with urlopen(request, timeout=60) as response:
        return response.read()


def discover(company_code):
    company, market, org = COMPANIES[company_code]
    form = dict(pageNum=1, pageSize=100, column=market, tabName="fulltext",
                plate="", stock=f"{company_code},{org}", searchkey="", secid="",
                category="category_ndbg_szsh;category_bndbg_szsh;", trade="",
                seDate="2024-07-01~2026-07-01", sortName="time", sortType="desc",
                isHLtitle="false")
    response = json.loads(read_url(f"{CNINFO}/new/hisAnnouncement/query", form))
    found = {}
    for row in response.get("announcements") or []:
        title = row["announcementTitle"].replace(" ", "")
        match = re.search(r"(202[45])年(年度|半年度)报告$", title)
        if row["secCode"] != company_code or not match:
            continue
        year, period = int(match[1]), "H1" if match[2] == "半年度" else "FY"
        key = (year, period)
        if key in found:
            continue
        published_at = datetime.fromtimestamp(row["announcementTime"] / 1000, CHINA).date()
        relative = row["adjunctUrl"]
        if not relative.lower().endswith(".pdf") or not relative.startswith("finalpage/"):
            continue
        found[key] = {
            "id": f"{company_code}-{year}-{period}", "company": company,
            "company_code": company_code, "year": year, "period": period,
            "published_at": published_at.isoformat(), "title": title,
            "url": urljoin(f"{STATIC}/", relative), "version": "公开全文",
            "announcement_id": row["announcementId"],
        }
    if len(found) != 4:
        raise ValueError(f"{company}: expected four full filings, found {list(found)}")
    return list(found.values())


def download(report):
    path = REPORTS / f"{report['id']}.pdf"
    if not path.exists():
        pdf = read_url(report["url"])
        if not pdf.startswith(b"%PDF-"):
            raise ValueError(f"Expected a PDF from {report['url']}")
        staging = path.with_suffix(".download")
        staging.write_bytes(pdf)
        staging.replace(path)
    content = path.read_bytes()
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"Invalid PDF: {path}")
    digest = sha256(content).hexdigest()
    if report.get("sha256", digest) != digest:
        raise ValueError(f"Checksum mismatch: {path}")
    return dict(report, sha256=digest, bytes=len(content))


def acquire():
    REPORTS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        if MANIFEST.exists():
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        else:
            manifest = [report for company in pool.map(discover, COMPANIES) for report in company]
        manifest = sorted(pool.map(download, manifest), key=lambda report: report["id"])
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for report in manifest:
        print(report["id"], report["published_at"], report["bytes"], report["url"])


if __name__ == "__main__":
    acquire()
