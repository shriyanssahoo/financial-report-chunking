"""Parse the subset PDFs into elements (Title / NarrativeText / Table ...) with Unstructured.

Writes one JSON file per document to data/parsed/<doc_name>.json, in the format
run_element.py expects. Already-parsed documents are skipped, so you can stop (Ctrl+C)
and re-run at any time.

Usage (from the repo root):
    python src/parse_pdfs.py                      # all subset docs, shortest first, hi_res
    python src/parse_pdfs.py --max-pages 100      # only docs up to 100 pages
    python src/parse_pdfs.py --docs AMD_2022_10K  # specific docs
    python src/parse_pdfs.py --strategy fast      # quick test (no table detection!)

hi_res downloads a layout-detection model from Hugging Face the first time and is slow on CPU
(expect several seconds per page). 'fast' is only for testing the pipeline: it never produces
Table elements, so it is NOT a fair element-chunking run.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pymupdf
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_dicts

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
PARSED = ROOT / "data" / "parsed"
DOCS_CSV = ROOT / "data" / "processed" / "subset_docs.csv"


def parse_one(doc, strategy, out_dir):
    pdf = PDF_DIR / f"{doc}.pdf"
    els = partition_pdf(
        filename=str(pdf),
        strategy=strategy,
        infer_table_structure=(strategy == "hi_res"),
    )
    dicts = elements_to_dicts(els)
    tmp = out_dir / f"{doc}.json.tmp"
    tmp.write_text(json.dumps(dicts))
    tmp.rename(out_dir / f"{doc}.json")   # only finished files carry the final name
    n_tables = sum(d["type"] == "Table" for d in dicts)
    return len(dicts), n_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="hi_res", choices=["hi_res", "fast", "auto"])
    ap.add_argument("--max-pages", type=int, default=10**9)
    ap.add_argument("--docs", nargs="*")
    ap.add_argument("--out", default=str(PARSED))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(DOCS_CSV).sort_values("n_pages")
    docs = docs[docs["n_pages"] <= args.max_pages]
    if args.docs:
        docs = docs[docs["doc_name"].isin(args.docs)]

    for _, r in docs.iterrows():
        doc = r["doc_name"]
        if (out_dir / f"{doc}.json").exists():
            print(f"SKIP {doc} (already parsed)")
            continue
        if not (PDF_DIR / f"{doc}.pdf").exists():
            print(f"MISSING PDF {doc} -> unzip subset_pdfs.zip into data/pdfs/")
            continue
        t = time.time()
        n, nt = parse_one(doc, args.strategy, out_dir)
        print(f"OK   {doc}: {r['n_pages']} pages, {n} elements, {nt} tables, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()