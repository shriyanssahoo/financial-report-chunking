import re
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT = ROOT / "data" / "processed"

MAX_PAGES = 200
N_DOCS = 20
SEED = 42

qa = pd.read_json(OUT / "questions_available.jsonl", lines=True)
ev = pd.read_json(OUT / "evidence_pieces.jsonl", lines=True)

# Page counts per document
docs = pd.DataFrame({"doc_name": sorted(qa["doc_name"].unique())})
docs["n_pages"] = docs["doc_name"].apply(
    lambda d: len(pymupdf.open(PDF_DIR / f"{d}.pdf"))
)
docs = docs.merge(qa.groupby("doc_name").size().rename("n_questions").reset_index())
print("All docs: pages summary")
print(docs["n_pages"].describe().round(1))

# Eligible docs, then a reproducible sample
eligible = docs[docs["n_pages"] <= MAX_PAGES]
print(f"\nEligible docs (<= {MAX_PAGES} pages): {len(eligible)} "
      f"with {eligible['n_questions'].sum()} questions")
chosen = eligible.sample(n=min(N_DOCS, len(eligible)), random_state=SEED)
chosen_names = set(chosen["doc_name"])

# Per-question gold pages (0-indexed) and a rough "numeric/table-like" flag
def digit_ratio(text):
    t = re.sub(r"\s", "", text or "")
    return sum(c.isdigit() for c in t) / max(len(t), 1)

ev["digit_ratio"] = ev["evidence_text"].apply(digit_ratio)
g = ev.groupby("financebench_id").agg(
    gold_pages=("evidence_page_num", lambda s: sorted({int(x) for x in s})),
    gold_texts=("evidence_text", list),
    max_digit_ratio=("digit_ratio", "max"),
).reset_index()
g["table_like"] = g["max_digit_ratio"] > 0.25   # heuristic only

sub = qa[qa["doc_name"].isin(chosen_names)].merge(g, on="financebench_id")
sub = sub.drop(columns=["evidence"])

print(f"\nSelected {sub['doc_name'].nunique()} docs, {len(sub)} questions")
print("\nQuestion types:\n", sub["question_type"].value_counts().to_string())
print("\nQuestion reasoning:\n", sub["question_reasoning"].value_counts().to_string())
print("\nTable-like evidence (heuristic):", int(sub["table_like"].sum()), "/", len(sub))
print("Questions with multiple gold pages:", int((sub["gold_pages"].str.len() > 1).sum()))

sub.to_json(OUT / "subset_questions.jsonl", orient="records", lines=True)
chosen.sort_values("doc_name").to_csv(OUT / "subset_docs.csv", index=False)
print("\nSaved subset_questions.jsonl and subset_docs.csv in", OUT)