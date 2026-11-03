"""Paired comparison of strategies on the SAME questions (no re-running, no API).

For two methods A and B it counts questions where A finds the gold page and B does not (+) and the
reverse (-), using the strict score (a chunk must START on a gold page). An exact sign test gives a
p-value. With ~42 questions, p < 0.05 is hard to reach: report the +/- counts, not only the p-value.

Run from the repo root:   python src/paired_stats.py
Writes results/retrieval/paired_stats.csv
"""
import glob
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/retrieval"
q = pd.read_json(ROOT / "data/processed/subset_questions.jsonl", lines=True).set_index("financebench_id")

files = [f for f in glob.glob(str(RES / "*_retrieval*.jsonl")) if "_test" not in f]
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
ids = set.intersection(*[set(g["financebench_id"]) for _, g in df.groupby("strategy")])
d = df[df["financebench_id"].isin(ids)].copy()
d["hit"] = d.apply(lambda r: bool(set(q.loc[r["financebench_id"], "gold_pages"]) &
                                  {min(c["pages"]) for c in r["retrieved"]}), axis=1)
piv = d.pivot(index="financebench_id", columns="strategy", values="hit")
print(f"{len(piv)} questions answered by all {piv.shape[1]} strategies\n")


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def find(prefix, suffix=""):
    return [c for c in piv.columns if c.startswith(prefix) and c.endswith(suffix)][0]


he, hl = find("hier_element_P10"), find("hier_llm_", "_P10")
lp = find("llm_", "_plain")
pairs = [
    ("Hierarchical vs flat (same element chunks)", he, "element_plain"),
    ("Hierarchical vs flat (same LLM chunks)", hl, lp),
    ("Hierarchical(element) vs element prefix", he, "element_prefix"),
    ("Hierarchical(element) vs base_128", he, "base_128"),
    ("Hierarchical(element) vs base_256", he, "base_256"),
    ("Hierarchical(element) vs base_512", he, "base_512"),
    ("Hierarchical(LLM) vs base_128", hl, "base_128"),
    ("Hierarchical(LLM) vs base_256", hl, "base_256"),
    ("Hierarchical(LLM) vs base_512", hl, "base_512"),
    ("LLM chunking vs element chunking (flat)", lp, "element_plain"),
]
rows = []
for name, a, b in pairs:
    w, l = int((piv[a] & ~piv[b]).sum()), int((~piv[a] & piv[b]).sum())
    rows.append({"comparison": name, "A": a, "B": b, "A_wins": w, "B_wins": l, "p_exact": round(sign_p(w, l), 3)})
out = pd.DataFrame(rows)
print(out[["comparison", "A_wins", "B_wins", "p_exact"]].to_string(index=False))
out.to_csv(RES / "paired_stats.csv", index=False)
print("\nSaved", RES / "paired_stats.csv")