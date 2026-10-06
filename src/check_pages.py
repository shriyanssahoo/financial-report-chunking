import re
from collections import Counter
from pathlib import Path

import pymupdf
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
OUT = ROOT / "data" / "processed"

qa = pd.read_json(OUT / "questions_available.jsonl", lines=True)

# ---- 1. Show the real schema -------------------------------------------
print("COLUMNS:", list(qa.columns))
row0 = qa.iloc[0].to_dict()
for k, v in row0.items():
    s = str(v)
    print(f"  {k}: {s[:150]}{'...' if len(s) > 150 else ''}")

# ---- 2. Unpack evidence into one row per piece --------------------------
rows = []
for _, r in qa.iterrows():
    ev = r["evidence"]
    if isinstance(ev, dict):
        ev = [ev]
    for i, e in enumerate(ev):
        rows.append({
            "financebench_id": r["financebench_id"],
            "doc_name": r["doc_name"],
            "question": r["question"],
            "answer": r["answer"],
            "question_type": r.get("question_type"),
            "question_reasoning": r.get("question_reasoning"),
            "evidence_idx": i,
            "evidence_text": e.get("evidence_text", ""),
            "evidence_page_num": e.get("evidence_page_num"),
        })
ev_df = pd.DataFrame(rows)
print(f"\nQuestions: {qa.shape[0]} | evidence pieces: {len(ev_df)}")
print("Questions with >1 evidence piece:",
      (ev_df.groupby('financebench_id').size() > 1).sum())

# ---- 3. Find which PDF page actually contains each evidence text --------
def words(text):
    return set(w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2)

page_cache = {}
def page_wordsets(doc_name):
    if doc_name not in page_cache:
        doc = pymupdf.open(PDF_DIR / f"{doc_name}.pdf")
        page_cache[doc_name] = [words(p.get_text()) for p in doc]
    return page_cache[doc_name]

def best_page(doc_name, evidence_text):
    ew = words(evidence_text)
    if len(ew) < 8:
        return None, 0.0
    best, best_score = None, 0.0
    for i, pw in enumerate(page_wordsets(doc_name)):
        score = len(ew & pw) / len(ew)
        if score > best_score:
            best, best_score = i, score
    return best, best_score

found, scores = [], []
for _, r in ev_df.iterrows():
    p, s = best_page(r["doc_name"], r["evidence_text"])
    found.append(p)
    scores.append(s)
ev_df["pdf_page_0idx"] = found
ev_df["match_score"] = scores

# Only trust confident matches
good = ev_df[(ev_df["match_score"] >= 0.8) & ev_df["pdf_page_0idx"].notna()].copy()
good["offset"] = good["pdf_page_0idx"] - good["evidence_page_num"].astype(int)

print(f"\nConfident matches: {len(good)} / {len(ev_df)}")
print("Offset = (true 0-indexed PDF page) - (dataset evidence_page_num):")
print(Counter(good["offset"]).most_common(8))

dominant, count = Counter(good["offset"]).most_common(1)[0]
print(f"\n>>> Dominant offset: {dominant} ({count}/{len(good)} = {count/len(good):.0%})")
if dominant == 0:
    print("Dataset page numbers are 0-indexed (they match PyMuPDF page indices directly).")
elif dominant == -1:
    print("Dataset page numbers are 1-indexed (subtract 1 to get PyMuPDF indices).")
else:
    print("Unusual offset: inspect the mismatches below.")

# Show a few disagreements so we can inspect them by hand
bad = good[good["offset"] != dominant]
print(f"\nPieces not matching the dominant offset: {len(bad)}")
print(bad[["doc_name", "evidence_page_num", "pdf_page_0idx", "match_score"]].head(10).to_string())

ev_df.to_json(OUT / "evidence_pieces.jsonl", orient="records", lines=True)
print("\nSaved:", OUT / "evidence_pieces.jsonl")