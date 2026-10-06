import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PDF_DIR = ROOT / "data" / "pdfs"
OUT = ROOT / "data" / "processed"
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{name}.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; student project)"}


def is_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def try_download(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200 and is_pdf(r.content):
                dest.write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return False


def main():
    qa = pd.read_json(RAW / "financebench_open_source.jsonl", lines=True)
    docs = pd.read_json(RAW / "financebench_document_information.jsonl", lines=True)

    # Only the documents referenced by the 150 benchmark questions
    needed = sorted(qa["doc_name"].unique())
    print(f"Questions: {len(qa)} | documents used by those questions: {len(needed)}")

    links = docs.drop_duplicates("doc_name").set_index("doc_name")["doc_link"].to_dict()

    ok_docs, failed = [], []
    for name in needed:
        dest = PDF_DIR / f"{name}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            ok_docs.append(name)
            print(f"SKIP {name} (already downloaded)")
            continue

        ok = try_download(GITHUB_RAW.format(name=name), dest)
        source = "github"
        link = links.get(name)
        if not ok and isinstance(link, str):
            ok = try_download(link, dest)
            source = "doc_link"

        if ok:
            ok_docs.append(name)
            print(f"OK   {name} ({source})")
        else:
            failed.append(name)
            print(f"FAIL {name}")

    # Keep only questions whose PDF is available (the paper did the same: 150 -> 141)
    available = qa[qa["doc_name"].isin(ok_docs)]
    available.to_json(OUT / "questions_available.jsonl", orient="records", lines=True)

    print("\n" + "=" * 50)
    print(f"Documents downloaded : {len(ok_docs)} / {len(needed)}   (paper: 80 / 84)")
    print(f"Questions usable     : {len(available)} / {len(qa)}   (paper: 141 / 150)")
    if failed:
        print("\nFailed documents (download manually from doc_link, save as <doc_name>.pdf):")
        for n in failed:
            print(f"  - {n}  ->  {links.get(n)}")


if __name__ == "__main__":
    main()