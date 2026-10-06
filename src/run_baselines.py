import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from chunkers import fixed_token_chunks, page_texts

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR, OUT = ROOT / "data/pdfs", ROOT / "data/processed"
IDX, RES = ROOT / "indexes", ROOT / "results/retrieval"
IDX.mkdir(exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SIZES = [128, 256, 512]
TOP_K = 10
QFILE = sys.argv[1] if len(sys.argv) > 1 else "subset_questions.jsonl"

enc = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
bleu = BLEU(effective_order=True)

qs = pd.read_json(OUT / QFILE, lines=True)


def best_overlap(retrieved_texts, gold_texts):
    best = {"p": 0.0, "r": 0.0, "f": 0.0, "bleu": 0.0}
    for g in gold_texts:
        for c in retrieved_texts:
            s = rouge.score(g, c)["rougeL"]
            best["p"] = max(best["p"], s.precision)
            best["r"] = max(best["r"], s.recall)
            best["f"] = max(best["f"], s.fmeasure)
            best["bleu"] = max(best["bleu"], bleu.sentence_score(c, [g]).score / 100)
    return best


records = []
for n in SIZES:
    strategy = f"base_{n}"
    n_chunks = {}
    for doc, dq in tqdm(qs.groupby("doc_name"), desc=strategy):
        cache = IDX / f"{doc}__{strategy}"
        if (cache.with_suffix(".npy")).exists():
            chunks = json.loads(cache.with_suffix(".json").read_text())
            vecs = np.load(cache.with_suffix(".npy"))
        else:
            chunks = fixed_token_chunks(page_texts(PDF_DIR / f"{doc}.pdf"), n)
            vecs = enc.encode([c["text"] for c in chunks], batch_size=32,
                              show_progress_bar=False)
            cache.with_suffix(".json").write_text(json.dumps(chunks))
            np.save(cache.with_suffix(".npy"), vecs)
        n_chunks[doc] = len(chunks)

        for _, q in dq.iterrows():
            qv = enc.encode([q["question"]])[0]
            top = np.argsort(-(vecs @ qv))[:TOP_K]          # exact dot-product search
            retrieved = [chunks[i] for i in top]
            gold_pages = set(q["gold_pages"])
            hit = any(gold_pages & set(c["pages"]) for c in retrieved)
            ov = best_overlap([c["text"] for c in retrieved], q["gold_texts"])
            records.append({
                "strategy": strategy, "financebench_id": q["financebench_id"],
                "doc_name": doc, "table_like": bool(q["table_like"]),
                "page_hit": hit, "rougeL_p": ov["p"], "rougeL_r": ov["r"],
                "rougeL_f": ov["f"], "bleu": ov["bleu"],
                "retrieved": [{"pages": c["pages"], "text": c["text"]} for c in retrieved],
            })
    print(f"{strategy}: {sum(n_chunks.values())} chunks over {len(n_chunks)} docs")

df = pd.DataFrame(records)
df.to_json(RES / "baseline_retrieval.jsonl", orient="records", lines=True)

summary = df.groupby("strategy").agg(
    questions=("page_hit", "size"),
    page_acc=("page_hit", lambda s: 100 * s.mean()),
    rougeL_f=("rougeL_f", "mean"), rougeL_r=("rougeL_r", "mean"),
    bleu=("bleu", "mean"),
).round(3)
print("\n", summary.to_string())
print("\nTable-like questions only:")
print(df[df["table_like"]].groupby("strategy")["page_hit"].mean().mul(100).round(1).to_string())
summary.to_csv(RES / "baseline_summary.csv")