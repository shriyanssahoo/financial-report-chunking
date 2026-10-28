"""Fairer comparison of all strategies, computed from the saved results (no re-running, no API).

Why: a fixed 512-token window often straddles two pages and the top-10 of them contain ~2x more text
than element chunks. 'page_any' (any page touched counts) therefore favours big fixed windows.
This script reports, on exactly the same questions for every strategy:
  page_any    - old metric: a hit if ANY page of ANY retrieved chunk is a gold page
  page_start  - stricter: only the page where each chunk STARTS counts
  chars       - average amount of text retrieved (top 10) -> compare strategies with similar chars
  rougeL_f    - text overlap with the gold evidence

Run from the repo root:   python src/fair_metrics.py
Writes results/retrieval/fair_summary.csv
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/retrieval"

qs = pd.read_json(ROOT / "data/processed/subset_questions.jsonl", lines=True).set_index("financebench_id")

files = [f for f in glob.glob(str(RES / "*_retrieval*.jsonl")) if "_test" not in f]
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

# keep only questions that EVERY strategy answered
n_strat = df["strategy"].nunique()
common = df.groupby("financebench_id")["strategy"].nunique()
common = common[common == n_strat].index
df = df[df["financebench_id"].isin(common)].copy()


def score(r):
    gold = set(qs.loc[r["financebench_id"], "gold_pages"])
    ch = r["retrieved"]
    return pd.Series({
        "page_any": bool(gold & set().union(*[set(c["pages"]) for c in ch])),
        "page_start": bool(gold & {min(c["pages"]) for c in ch}),
        "chars": sum(len(c["text"]) for c in ch),
    })


df = pd.concat([df.reset_index(drop=True), df.apply(score, axis=1).reset_index(drop=True)], axis=1)

summary = df.groupby("strategy").agg(
    questions=("page_any", "size"),
    page_any=("page_any", lambda s: 100 * s.mean()),
    page_start=("page_start", lambda s: 100 * s.mean()),
    chars=("chars", "mean"),
    rougeL_f=("rougeL_f", "mean"),
).round(2)
tl = df[df["table_like"]].groupby("strategy")["page_start"].mean().mul(100).round(1).rename("table_page_start")
summary = summary.join(tl)
print(f"{len(common)} questions answered by all {n_strat} strategies\n")
print(summary.to_string())
summary.to_csv(RES / "fair_summary.csv")
print("\nSaved", RES / "fair_summary.csv")