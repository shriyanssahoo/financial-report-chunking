import re
from pathlib import Path

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
RAW, PDF_DIR, OUT = ROOT / "data/raw", ROOT / "data/pdfs", ROOT / "data/processed"

MAX_PAGES_PER_DOC = 300
PAGE_BUDGET = 3000   # total pages we are willing to parse with hi_res

qa = pd.read_json(OUT / "questions_available.jsonl", lines=True)
ev = pd.read_json(OUT / "evidence_pieces.jsonl", lines=True)
meta = (pd.read_json(RAW / "financebench_document_information.jsonl", lines=True)
        .drop_duplicates("doc_name")[["doc_name", "doc_type"]])

docs = pd.DataFrame({"doc_name": sorted(qa["doc_name"].unique())})
docs["n_pages"] = docs["doc_name"].apply(lambda d: len(pymupdf.open(PDF_DIR / f"{d}.pdf")))
docs = (docs.merge(qa.groupby("doc_name").size().rename("n_questions").reset_index())
            .merge(meta, on="doc_name", how="left"))
docs["density"] = docs["n_questions"] / docs["n_pages"]

# Greedy: densest documents first until the page budget is used
chosen, used = [], 0
for _, r in docs[docs["n_pages"] <= MAX_PAGES_PER_DOC].sort_values("density", ascending=False).iterrows():
    if used + r["n_pages"] <= PAGE_BUDGET:
        chosen.append(r["doc_name"])
        used += r["n_pages"]
chosen_docs = docs[docs["doc_name"].isin(chosen)].sort_values("doc_name")

def digit_ratio(text):
    t = re.sub(r"\s", "", text or "")
    return sum(c.isdigit() for c in t) / max(len(t), 1)

ev["digit_ratio"] = ev["evidence_text"].apply(digit_ratio)
g = ev.groupby("financebench_id").agg(
    gold_pages=("evidence_page_num", lambda s: sorted({int(x) for x in s})),
    gold_texts=("evidence_text", list),
    max_digit_ratio=("digit_ratio", "max"),
).reset_index()
g["table_like"] = g["max_digit_ratio"] > 0.25   # rough heuristic only

sub = qa[qa["doc_name"].isin(chosen)].merge(g, on="financebench_id").drop(columns=["evidence"])

print(f"Selected {len(chosen_docs)} docs | {used} pages | {len(sub)} questions")
print("\nDocument types (selected):\n", chosen_docs["doc_type"].value_counts().to_string())
print("\nDocument types (all 84):\n", docs["doc_type"].value_counts().to_string())
print("\nQuestion types:\n", sub["question_type"].value_counts().to_string())
print("\nTable-like evidence:", int(sub["table_like"].sum()), "/", len(sub))
print("Multi-page gold:", int((sub["gold_pages"].str.len() > 1).sum()))

sub.to_json(OUT / "subset_questions.jsonl", orient="records", lines=True)
chosen_docs.to_csv(OUT / "subset_docs.csv", index=False)