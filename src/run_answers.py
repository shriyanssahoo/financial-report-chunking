"""End-to-end test: can an LLM ANSWER the question from what each method retrieved?

For every question and every method we give the LLM the question + the 10 retrieved chunks, get an
answer, and let the LLM judge it against FinanceBench's gold answer. A 4th row, 'oracle', gets the gold
evidence text instead of retrieved chunks: it shows how well the LLM does when retrieval is perfect.

Needs only files you already have (results/retrieval/*.jsonl) and the Gemini key. No parsing, no embeddings.
Resumable: every answer/verdict is saved at once; re-running skips what is done.

PowerShell, from the repo root (same window where the key is set):
    $env:LLM_API_KEY = "<your NEW gemini key>"
    python src/run_answers.py --limit 3          # tiny test first (3 questions)
    python src/run_answers.py                     # the full run
Results: results/answers/answers.jsonl, answers_summary.csv, check_these.csv
"""
import argparse
import json
import math
import os
import random
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(os.environ.get("ANS_ROOT") or Path(__file__).resolve().parent.parent)
RES, OUT = ROOT / "results/retrieval", ROOT / "results/answers"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
ORACLE_CHARS = 12000

# (name shown in the table, file, strategy name inside the file)
METHODS = [
    ("base_256", "baseline_retrieval.jsonl", "base_256"),
    ("hier_element_P10", "hier_retrieval__element.jsonl", "hier_element_P10"),
    ("hier_llm_P10", "hier_retrieval__llm_gemini-3.5-flash-lite.jsonl", "hier_llm_gemini-3.5-flash-lite_P10"),
]

ANSWER_PROMPT = (
    "You are answering a question about a company's financial report. Use ONLY the excerpts below. "
    "If the excerpts do not contain the information, say so briefly instead of guessing. "
    "Show the key numbers you used, then give a short final answer.\n\n"
    "EXCERPTS:\n{ctx}\n\nQUESTION: {q}\n\nANSWER:")

JUDGE_PROMPT = (
    "You are grading an answer to a financial question against a reference answer.\n"
    "Question: {q}\nReference answer: {gold}\nCandidate answer: {ans}\n\n"
    "Is the candidate's FINAL answer consistent with the reference answer? Rules: the key number or fact "
    "must match (rounding and formatting differences such as $4.3 billion vs 4,300 million are fine; "
    "different numbers or the opposite conclusion are wrong). An answer that says the information is "
    "missing is wrong. Extra explanation does not matter.\n"
    'Reply with ONLY JSON: {{"correct": true}} or {{"correct": false}}')

lock = threading.Lock()
last_call = [0.0]


class Quota(RuntimeError):
    pass


def llm(prompt, args):
    """One chat call with throttling and retries. Raises Quota/RuntimeError when the run must stop."""
    key = os.environ.get("LLM_API_KEY", "")
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    for attempt in range(6):
        if args.rpm:
            wait = last_call[0] + 60.0 / args.rpm - time.time()
            if wait > 0:
                time.sleep(wait)
            last_call[0] = time.time()
        try:
            r = requests.post(f"{BASE_URL.rstrip('/')}/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {key}"}, json=body)
        except requests.RequestException as e:
            time.sleep(min(5 * 2 ** attempt, 60))
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"] or ""
        if r.status_code in (401, 403, 404):
            raise RuntimeError(f"The API refused the request ({r.status_code}): {r.text[:200]}. "
                               "Check $env:LLM_API_KEY (set in THIS window) and --model.")
        if r.status_code == 400:
            return ""
        wait = min(5 * 2 ** attempt, 60)
        if r.status_code == 429:
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", r.text)
            if m:
                wait = int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
            if wait > 180:
                raise Quota("Daily limit reached. Re-run later: finished answers are saved and are not asked again.")
        time.sleep(wait)
    raise Quota("Too many failed calls in a row (rate limit?). Re-run in a few minutes: progress is saved.")


def parse_correct(raw):
    m = re.search(r'"correct"\s*:\s*(true|false)', raw or "", re.I)
    return None if not m else m.group(1).lower() == "true"


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--rpm", type=int, default=12, help="max calls per minute (0 = no limit)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (for a quick test)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    qs = pd.read_json(ROOT / "data/processed/subset_questions.jsonl", lines=True).set_index("financebench_id")

    ret = {}
    for name, fname, strat in METHODS:
        d = pd.read_json(RES / fname, lines=True)
        d = d[d["strategy"] == strat]
        if d.empty:
            raise SystemExit(f"No rows for {strat} in {fname}. Is the file in results/retrieval/ ?")
        ret[name] = {r["financebench_id"]: r["retrieved"] for _, r in d.iterrows()}
    ids = sorted(set.intersection(*[set(v) for v in ret.values()]))
    if args.limit:
        ids = ids[:args.limit]
    methods = [m[0] for m in METHODS] + ["oracle"]
    print(f"{len(ids)} questions x {len(methods)} methods = {len(ids) * len(methods)} answers "
          f"(+ as many judge calls). Rate limit {args.rpm}/min.")

    f_out = OUT / "answers.jsonl"
    done = {}
    if f_out.exists():
        for line in open(f_out, encoding="utf-8"):
            r = json.loads(line)
            done[(r["method"], r["financebench_id"])] = r
    print(f"{len(done)} already finished, skipping them.")

    t0, n_new = time.time(), 0
    try:
        for qid in ids:
            q = qs.loc[qid]
            for m in methods:
                if (m, qid) in done and done[(m, qid)].get("correct") is not None:
                    continue
                if m == "oracle":
                    ctx = "\n\n".join(q["gold_texts"])[:ORACLE_CHARS]
                else:
                    ctx = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(ret[m][qid], 1))
                ans = llm(ANSWER_PROMPT.format(ctx=ctx, q=q["question"]), args).strip()
                raw = llm(JUDGE_PROMPT.format(q=q["question"], gold=q["answer"], ans=ans[:3000]), args)
                rec = {"method": m, "financebench_id": qid, "table_like": bool(q["table_like"]),
                       "answer": ans, "judge_raw": raw, "correct": parse_correct(raw)}
                done[(m, qid)] = rec
                with open(f_out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_new += 1
                if n_new % 10 == 0:
                    print(f"  {n_new} new answers, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
    except (Quota, RuntimeError) as e:
        print(f"\nSTOPPED: {e}\nSummary below covers what is finished so far.")

    # ---------------------------------------------------------------- summary
    rows = [r for r in done.values() if r["correct"] is not None]
    df = pd.DataFrame(rows).drop_duplicates(["method", "financebench_id"], keep="last")
    piv = df.pivot(index="financebench_id", columns="method", values="correct")
    full = piv.dropna()
    print(f"\n{len(full)} questions finished for ALL methods (table uses only these).")
    if full.empty:
        return
    tl = df.drop_duplicates("financebench_id").set_index("financebench_id")["table_like"].reindex(full.index)
    full = full.astype(bool)
    out = []
    for m in methods:
        out.append({"method": m, "questions": len(full), "correct": int(full[m].sum()),
                    "accuracy_%": round(100 * full[m].mean(), 1),
                    "table_like_%": round(100 * full.loc[tl, m].mean(), 1) if tl.any() else None})
    summ = pd.DataFrame(out)
    print(summ.to_string(index=False))
    print("\nPaired comparison (questions where only one of the two is correct):")
    pairs = [("hier_llm_P10", "base_256"), ("hier_element_P10", "base_256"), ("hier_llm_P10", "hier_element_P10")]
    for a, b in pairs:
        w, l = int((full[a] & ~full[b]).sum()), int((~full[a] & full[b]).sum())
        print(f"  {a} vs {b}: {a} only {w}, {b} only {l}, p = {sign_p(w, l):.3f}")
        summ.loc[len(summ)] = {"method": f"{a} vs {b}", "questions": len(full), "correct": f"{w}-{l}",
                               "accuracy_%": f"p={sign_p(w, l):.3f}", "table_like_%": None}
    summ.to_csv(OUT / "answers_summary.csv", index=False)

    random.seed(0)                                       # a fixed sample for the hand check
    sample = random.sample(list(full.index), min(10, len(full)))
    chk = []
    for qid in sample:
        for m in methods:
            r = done[(m, qid)]
            chk.append({"financebench_id": qid, "method": m, "question": qs.loc[qid, "question"],
                        "gold_answer": qs.loc[qid, "answer"], "model_answer": r["answer"][:600],
                        "judge_says_correct": r["correct"], "your_verdict": ""})
    pd.DataFrame(chk).to_csv(OUT / "check_these.csv", index=False)
    print(f"\nSaved {OUT / 'answers_summary.csv'} and {OUT / 'check_these.csv'} (10 questions to verify by eye).")


if __name__ == "__main__":
    main()