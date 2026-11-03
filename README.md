# Chunking Financial Reports for Retrieval-Augmented Generation (RAG)

### Baselines, Element-Based, LLM-Based and Hierarchical Retrieval

**Institution:** Indian Institute of Information Technology (IIIT), Dharwad
**Course:** LLM for Financial Applications

---

## 1. Introduction

Large financial documents, such as annual reports, quarterly reports and earnings releases, contain hundreds of pages of information.

Suppose someone asks an AI system:

> What was Apple's total revenue in 2023?

The AI needs to locate the correct information in Apple's financial reports before generating an answer.

This is where **Retrieval-Augmented Generation (RAG)** comes into play.

A RAG system generally works in two steps:

1. **Retrieval:** Find the most relevant information from a collection of documents.
2. **Generation:** Pass the retrieved information to an LLM, which uses it to generate an answer.

However, retrieving the correct information from large documents is not always easy.

One important factor is **chunking**.

Chunking means dividing a large document into smaller pieces of text called chunks. These chunks are converted into embeddings and stored or indexed for retrieval.

The way we divide a document can significantly affect which information the retrieval system finds.

### What are we trying to achieve?

Our project investigates different chunking and retrieval strategies to understand:

- Which chunking method retrieves the most relevant information?
- Does understanding document structure improve retrieval?
- Can an LLM identify better chunk boundaries than traditional rules?
- Can hierarchical retrieval improve the accuracy of document search?
- Does better retrieval lead to more correct answers?
- How do these methods perform on financial tables and numerical questions?

We first implemented existing approaches from a research paper and then explored two improvements suggested by our professor:

1. LLM-based chunking.
2. Hierarchical retrieval.

---

## 2. Dataset: FinanceBench

We use the **FinanceBench** dataset, which contains questions based on real-world financial documents.

Each question comes with:

- A question about a company's financial information.
- The corresponding document.
- The expected answer.
- Gold evidence text.
- The page number containing the evidence.

### Dataset statistics

| Parameter                     |                              Value |
| ----------------------------- | ---------------------------------: |
| Original questions            |                                150 |
| Documents selected            |                                 22 |
| Questions used in experiments |                                 42 |
| Maximum document length       |                          100 pages |
| Financial document types      | 10-K, 10-Q, 8-K, Earnings Releases |

We initially worked with the available FinanceBench questions and selected a subset of documents containing at most 100 pages because PDF parsing and processing can be time-consuming.

All the chunking and retrieval methods were evaluated on the same 42 questions to ensure a consistent comparison. The list of questions is in `data/processed/used_questions.csv`.

**Important:** Our 42-question subset comes from FinanceBench. It is not the exact evaluation subset used in the original research paper.

We also verified the dataset's page numbering against the actual PDF pages. The dataset's page numbers were found to be 0-indexed in our checks.

---

## 3. Overall Workflow

Our experimental pipeline follows these steps:

1. Download financial reports in PDF format.
2. Extract text and structural elements from the PDFs.
3. Apply different chunking techniques.
4. Convert chunks into embedding vectors.
5. Convert questions into embedding vectors.
6. Retrieve the top 10 most relevant chunks.
7. Compare the retrieved information with the original gold evidence.
8. Calculate evaluation metrics.
9. Give the retrieved chunks to an LLM, let it answer the question, and grade the answers.
10. Compare the performance of different approaches.

### Embedding Model

We use:

`multi-qa-mpnet-base-dot-v1`

This model converts text into numerical vectors (embeddings).

We calculate the similarity between question embeddings and chunk embeddings using **dot-product similarity**.

The chunks with the highest similarity scores are considered the most relevant.

---

## 4. Methodologies Implemented

We experimented with four major approaches.

### 4.1 Fixed-Size Chunking (Baseline)

This is the simplest chunking technique.

The document is divided into chunks containing a fixed number of tokens, without considering the document's structure.

We experimented with three chunk sizes:

| Configuration | Chunk Size |
| ------------- | ---------: |
| Baseline 1    | 128 tokens |
| Baseline 2    | 256 tokens |
| Baseline 3    | 512 tokens |

For example, a document containing 10,000 tokens is divided into consecutive chunks of the selected size.

**Limitation:**

Fixed-size chunking does not understand the meaning or structure of a document.

For example, a heading and its corresponding paragraph might end up in different chunks. A table could also get separated from its surrounding explanation.

Nevertheless, it provides an important baseline against which more advanced techniques can be compared.

### 4.2 Element-Based Chunking

This approach considers the structure of a PDF instead of simply dividing it into equal-sized pieces.

We use the **Unstructured library with the `hi_res` strategy** to identify different elements in financial documents.

These elements include:

- Titles
- Paragraphs
- Tables
- Other text elements

Our implementation follows these rules:

1. Whenever a new title is encountered, a new chunk is started.
2. Tables are preserved as separate chunks.
3. Other text elements are combined until approximately 2,048 characters.
4. Headers, footers and page numbers are excluded.

We also implemented a prefix variation.

**Element-Based Chunking with Prefix:**

The first two sentences of a chunk are added at the beginning of that chunk.

The purpose is to provide additional context that might help the embedding model understand the content better.

We evaluated both:

- Element (Plain)
- Element (Prefix)

### 4.3 LLM-Based Chunking

This was the first improvement suggested by our professor.

Instead of relying entirely on predefined rules to identify chunk boundaries, we use an LLM to identify changes in topics.

We use **Gemini 3.5 Flash-Lite** for this experiment.

#### How does it work?

1. The PDF is first parsed into individual elements.
2. The elements belonging to each page are numbered.
3. These numbered elements are provided to the LLM.
4. The LLM is asked to identify which element numbers represent the beginning of a new topic.
5. Our Python code uses those element numbers to divide the original text into chunks.

For example, the LLM might return:

`[1, 4, 9]`

This means that elements 1, 4 and 9 should begin new chunks.

**An important implementation detail:**

The LLM does not rewrite, summarize or modify the original document.

It only identifies the boundaries between topics. The actual chunk creation is performed by our Python code.

This helps preserve the original text and maintain accurate page tracking.

Tables are also kept separate.

We implemented additional features:

- Response caching to avoid unnecessary API calls.
- Retry mechanisms for failed requests.
- Fallback rules when the LLM response is unusable.

In our final experiment, there were zero fallback pages.

We evaluated:

- LLM Chunking (Plain)
- LLM Chunking (Prefix)

The prefix variation follows the same first-two-sentences approach used in element-based chunking.

### 4.4 Hierarchical Retrieval

This was the second improvement suggested by our professor.

Unlike the previous approaches, hierarchical retrieval changes how we search for information.

Instead of directly searching through every chunk in a document, we perform retrieval in two levels.

Think of it like finding a particular book in a library.

First, we identify the relevant section of the library. Then, we search for the specific book inside that section.

#### Level 1: Page-Level Retrieval

Each page of a financial document is represented by a short textual digest (the first 150 characters of each element on the page).

This representation is converted into an embedding vector.

When a question is received:

1. The question is converted into an embedding.
2. Its similarity with all page embeddings is calculated.
3. The top P most relevant pages are selected.

We experimented with:

- P = 5
- P = 10
- P = 15
- P = 20

#### Level 2: Chunk-Level Retrieval

After identifying the relevant pages:

1. We select only the chunks belonging to those pages.
2. We calculate the similarity between the question and these chunks.
3. We retrieve the top 10 chunks.

This reduces the search space and potentially improves retrieval accuracy.

We applied hierarchical retrieval to both:

- Element-based chunks.
- LLM-based chunks.

Therefore, we tested two combinations:

1. Hierarchical Retrieval + Element-Based Chunking.
2. Hierarchical Retrieval + LLM-Based Chunking.

---

## 5. Evaluation Metrics

We use multiple evaluation metrics to understand how well each approach performs.

### 5.1 Page Accuracy (Any)

A question is considered successfully retrieved if at least one of the retrieved chunks contains the gold evidence page.

This is the more permissive page-based metric.

### 5.2 Strict Page Score

A question is considered successful only if at least one retrieved chunk starts on the gold evidence page.

We introduced this stricter metric because fixed-size chunks can sometimes span multiple pages.

For example, a 512-token chunk might contain text from two different pages. Simply counting any overlap with the gold page could make larger chunks appear better than they actually are.

All results in this README use the strict score unless stated otherwise.

### 5.3 Text Retrieved

This measures the average number of characters returned in the top 10 retrieved chunks.

It helps us understand how much information each method retrieves.

A method that achieves similar accuracy while retrieving less text can be more efficient.

### 5.4 ROUGE-L F1 Score

ROUGE-L measures the overlap between the retrieved text and the original gold evidence.

A higher score indicates greater textual similarity.

### 5.5 BLEU Score

BLEU is another text-overlap metric used to compare retrieved text against the gold evidence.

### 5.6 Table Question Accuracy

We separately evaluate questions whose evidence is primarily numerical or table-based.

This helps us understand how well different methods retrieve information from financial tables.

Our evaluation subset contains 11 such questions.

### 5.7 Answer Accuracy

We give Gemini 3.5 Flash-Lite the question and the 10 chunks retrieved by a method, ask it to answer using only those chunks, and then ask the same model to grade the answer against FinanceBench's reference answer (the key number or fact must match; rounding and formatting differences are accepted; "the excerpts do not contain this" counts as wrong).

As a reference point we add an **oracle** row, where the model is given the gold evidence text instead of retrieved chunks. This shows how well the model performs when retrieval is perfect.

---

## 6. Experimental Results

All the following results are based on the same 42 questions.

### 6.1 Retrieval Results

| Method                        | Strict Page Score | Avg. Text Retrieved (Characters) | ROUGE-L F1 | Table Question Accuracy |
| ----------------------------- | ----------------: | -------------------------------: | ---------: | ----------------------: |
| Fixed 128                     |             83.3% |                            4,983 |       0.42 |                   63.6% |
| Fixed 256                     |             85.7% |                            9,745 |       0.49 |                   81.8% |
| Fixed 512                     |             78.6% |                           19,370 |       0.49 |                   81.8% |
| Element (Plain)               |             73.8% |                            8,686 |       0.49 |                   45.5% |
| Element (Prefix)              |             76.2% |                            9,571 |       0.49 |                   54.5% |
| LLM Chunking (Plain)          |             78.6% |                            6,066 |       0.48 |                   45.5% |
| LLM Chunking (Prefix)         |             73.8% |                            6,976 |       0.50 |                   45.5% |
| Hierarchical + Element (P=10) |             85.7% |                            7,812 |       0.51 |                   72.7% |
| **Hierarchical + LLM (P=10)** |         **88.1%** |                        **5,894** |   **0.52** |               **72.7%** |

Hierarchical + LLM has the highest score in this table, but its lead over Fixed 256 is one question (88.1% vs 85.7%) and is not statistically meaningful (Section 8). A fair summary is that it **matches the best fixed-size baseline while retrieving about 40% less text** (5,894 vs 9,745 characters). Fixed 128 retrieves less text still (4,983) but scores lower (83.3%).

### 6.2 Effect of Different P Values

We also investigated how the number of pages selected at Level 1 affects hierarchical retrieval.

| Strict Page Score      | P = 5 | P = 10 | P = 15 | P = 20 |
| ---------------------- | ----: | -----: | -----: | -----: |
| Hierarchical + Element | 71.4% |  85.7% |  78.6% |  76.2% |
| Hierarchical + LLM     | 73.8% |  88.1% |  83.3% |  81.0% |

**Observation:**

P = 10 performed best in our experiments.

Selecting too few pages can exclude the page containing the required evidence.

Selecting too many pages increases the search space and gradually reduces the advantage of hierarchical retrieval.

Note that P = 10 was chosen after looking at these results (see Limitations).

### 6.3 Answer Accuracy (End-to-End)

| Method (top-10 chunks given to the LLM) | Correct answers (of 42) | Accuracy | Table-like questions correct (of 11) |
| --------------------------------------- | ----------------------: | -------: | -----------------------------------: |
| Fixed 256                               |                      23 |    54.8% |                            4 (36.4%) |
| Hierarchical + Element (P=10)           |                      23 |    54.8% |                            4 (36.4%) |
| Hierarchical + LLM (P=10)               |                      25 |    59.5% |                            5 (45.5%) |
| Oracle (gold evidence given directly)   |                      27 |    64.3% |                           10 (90.9%) |

Paired comparison (questions where only one method answered correctly):

| Comparison                                   | A only | B only | p-value |
| -------------------------------------------- | -----: | -----: | ------: |
| Hierarchical + LLM vs Fixed 256              |      6 |      4 |   0.754 |
| Hierarchical + Element vs Fixed 256          |      5 |      5 |   1.000 |
| Hierarchical + LLM vs Hierarchical + Element |      2 |      0 |   0.500 |

**Interpretation:**

- All three retrieval methods give statistically indistinguishable answer accuracy. Hierarchical + LLM answers two more questions than Fixed 256, which is within noise.
- Combined with Section 6.1, the supported claim is: hierarchical retrieval with LLM chunks gives **the same answer quality as the best fixed-size baseline using about 40% less retrieved text.**
- **Table questions are the main weakness.** With the gold evidence the model answers 10 of 11 table-like questions, but with any retrieval method only 4 or 5. Fixed 256 found the right page for 81.8% of these questions (Section 6.1), so the right page is often retrieved but the chunk the model sees does not contain the needed part of the table. Page accuracy therefore overstates how useful retrieval is for numerical questions.
- The oracle is not a strict ceiling. It scores 64.3% because the model or the grader sometimes fails even with the gold evidence (for example, one question needs Adjusted EBIT, which is not in the gold evidence text), and on the 31 non-table questions Fixed 256 answered 19 correctly versus 17 for the oracle. Single-question differences should not be over-read.

**Grader check:** for 10 questions (40 answers) we compared the LLM judge with our own manual review. They agreed on 39 of 40 answers. The one disagreement (a Hierarchical + LLM answer on an MGM revenue question, where the main answer named the wrong segment and only hedged toward the correct one) was a case where the judge was slightly lenient. The reviewed rows are in `results/answers/check_these.csv`.

### 6.4 Additional Ideas Tried and Not Kept

We also tested three cheap add-ons to hierarchical retrieval on both chunk types, in all 8 combinations at P = 5 and P = 10: contextual headers (document name and section heading added before embedding), hybrid search (BM25 + dense embeddings combined with reciprocal-rank fusion), and pooled page vectors (page embedding = mean of its chunk embeddings).

- None of them improved strict page accuracy. Contextual headers and hybrid search left it unchanged for element chunks and lowered it by 1 to 4 questions for LLM chunks.
- Pooled page vectors hurt in every combination (for example, 85.7% to 69.0% for element chunks at P = 10).
- Headers plus hybrid search raised ROUGE-L F1 (0.51 to 0.57 for element chunks, 0.52 to 0.59 for LLM chunks) without improving page accuracy, at the cost of retrieving more text. We did not test this gain statistically.

We did not keep these experiments in the repository, and they are not part of the reported results.

---

## 7. Key Findings and Observations

### Finding 1: Fixed-Size Chunking Performs Surprisingly Well

We initially expected more sophisticated chunking approaches to outperform fixed-size chunking.

However, the results show that fixed 128-token and 256-token chunking performed better than standalone element-based and LLM-based chunking in terms of strict page accuracy.

This demonstrates that more advanced chunking does not automatically guarantee better retrieval. (We were also unable to reproduce an advantage for element-based chunking over fixed-size chunking on our data.)

### Finding 2: Element-Based Chunking Has Ranking Challenges

Element-based chunking performed relatively poorly, particularly on table-based questions.

We investigated the questions that it failed to retrieve.

Out of the 10 missed questions:

- 9 had the correct information properly extracted and present in the chunks.
- Only 1 had information lost during PDF parsing.

This suggests that the major problem was not necessarily extracting the information but ranking the relevant chunks high enough during retrieval.

### Finding 3: LLM-Based Chunking Alone Did Not Produce Major Improvements

LLM-based chunking helped identify more meaningful topic boundaries.

However, it did not significantly improve retrieval accuracy when used independently.

It did retrieve less text than most fixed-size baselines, but its standalone accuracy (78.6%) remained lower than the best fixed-size approaches.

### Finding 4: Hierarchical Retrieval Improved on Its Own Flat Version

This was the most consistent result of our experiments.

With P = 10, compared with the same chunks searched without the page level:

- Element-based retrieval improved from 73.8% to 85.7% (5 questions gained, 0 lost).
- LLM-based retrieval improved from 78.6% to 88.1% (4 gained, 0 lost).
- Table-question accuracy improved from 45.5% to 72.7% for both configurations.

Fixed 256 and Fixed 512 still score higher on table questions (81.8%), so hierarchical retrieval narrows but does not close that gap.

This suggests that identifying relevant pages before searching individual chunks can help.

### Finding 5: Similar Accuracy with Less Retrieved Text

Hierarchical + LLM (88.1%) matched Fixed 256 (85.7%) while retrieving about 40% less text (5,894 vs 9,745 characters). It is also far more compact than Fixed 512 (19,370 characters, 78.6%).

The same holds for answer accuracy (Section 6.3): no meaningful difference between methods, with less text sent to the LLM.

### Finding 6: Retrieving the Right Page Is Not the Same as Retrieving the Right Content

On table-like questions the model answers 10 of 11 with the gold evidence but only 4 to 5 with any retrieval method, even where the right page was found. Improving chunking and ranking for tables is the clearest remaining problem.

---

## 8. Paired Statistical Analysis

In addition to comparing overall accuracy, we examined the question-by-question differences between methods (strict page score, exact sign test; A wins = questions A retrieved correctly and B did not).

| Comparison                                     | A Wins | B Wins | p-value |
| ---------------------------------------------- | -----: | -----: | ------: |
| Hierarchical vs Flat (same element chunks)     |      5 |      0 |   0.062 |
| Hierarchical vs Flat (same LLM chunks)         |      4 |      0 |   0.125 |
| Hierarchical (Element) vs Element (Prefix)     |      4 |      0 |   0.125 |
| Hierarchical (Element) vs Fixed 128            |      4 |      3 |   1.000 |
| Hierarchical (Element) vs Fixed 256            |      4 |      4 |   1.000 |
| Hierarchical (Element) vs Fixed 512            |      7 |      4 |   0.549 |
| Hierarchical (LLM) vs Fixed 128                |      4 |      2 |   0.688 |
| Hierarchical (LLM) vs Fixed 256                |      3 |      2 |   1.000 |
| Hierarchical (LLM) vs Fixed 512                |      6 |      2 |   0.289 |
| LLM chunking (flat) vs Element chunking (flat) |      2 |      0 |   0.500 |

**Interpretation:**

Hierarchical retrieval never lost a question to its own flat version, which is directionally consistent. Against the fixed-size baselines it is statistically tied.

No comparison reaches the conventional 0.05 threshold, so improvements should be described as promising and directionally consistent, not statistically significant. The small evaluation set of 42 questions limits the strength of any conclusion (one question is about 2.4 percentage points, and the uncertainty on each accuracy is roughly +/-11 points).

---

## 9. Limitations

1. **Small Evaluation Dataset:** We evaluated only 42 questions. Larger datasets are required to establish stronger conclusions.

2. **Limited Document Length:** We selected documents containing at most 100 pages. The results may differ for longer financial reports.

3. **Limited Models:** We used one embedding model and one LLM (Gemini 3.5 Flash-Lite) for all reported experiments.

4. **LLM Judge for Answer Accuracy:** The same model that generates the answers also grades them. We checked 40 answers manually (39 agreed), but this is a small sample, and the answer-accuracy results come from a single run without repeated trials.

5. **Element-Based Implementation:** Our implementation follows the rules described in the research paper but uses Unstructured for PDF parsing rather than the paper's Chipper-based approach.

6. **Parameter Selection:** P = 10 performed best, but this value was selected after examining the experimental results. Therefore, the results should be interpreted with this limitation in mind.

7. **Multiple Configurations Tried:** Besides the methods reported here we tried 16 add-on configurations (Section 6.4). They are reported as a negative result, not as part of the main comparison.

---

## 10. Future Improvements

These are potential experiments and should not be confused with methods already implemented.

| Proposed Improvement      | Description                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| Better Table Retrieval    | Keep whole tables together with their captions and units, or index table rows separately.   |
| LLM Query Rewriting       | Rewrite questions using relevant financial terminology before retrieval.                    |
| Reranking                 | Retrieve more candidate chunks and use a stronger model to rerank them.                     |
| Larger Evaluation Dataset | Evaluate the approaches on more FinanceBench questions and longer reports.                  |
| More Models               | Repeat the experiments with other embedding models and LLMs, and with an independent judge. |

---

## 11. Repository Structure

```text
financial-report-chunking/
│
├── data/
│   ├── raw/
│   │   └── Original FinanceBench dataset
│   │
│   └── processed/
│       ├── Processed questions and document metadata
│       └── used_questions.csv        (the 42 questions used)
│
├── src/
│   ├── download_pdfs.py
│   ├── select_subset.py
│   ├── check_pages.py
│   ├── parse_pdfs.py
│   ├── chunkers.py
│   ├── run_baselines.py
│   ├── run_element.py
│   ├── run_llm.py
│   ├── run_hier.py
│   ├── run_answers.py                (answer generation and grading)
│   ├── fair_metrics.py
│   ├── paired_stats.py
│   ├── diagnose_element.py
│   └── zip_subset.py
│
├── results/
│   ├── retrieval/
│   │   └── Retrieval results, fair_summary.csv, paired_stats.csv
│   └── answers/
│       └── answers.jsonl, answers_summary.csv, check_these.csv
│
├── subset_pdfs.zip
├── requirements.txt
└── README.md
```

**Note:** Generated intermediate files, PDF parsing outputs, embedding indexes and LLM caches are not included in the repository.

### How to reproduce (from the repository root)

```text
python src/parse_pdfs.py --strategy hi_res --max-pages 100
python src/run_baselines.py
python src/run_element.py
python src/run_llm.py --model gemini-3.5-flash-lite --max-pages 100     # needs LLM_API_KEY
python src/run_hier.py --chunks element --max-pages 100
python src/run_hier.py --chunks llm --model gemini-3.5-flash-lite --max-pages 100
python src/fair_metrics.py
python src/paired_stats.py
python src/run_answers.py                                              # needs LLM_API_KEY
```

API keys are read from the `LLM_API_KEY` environment variable and are never stored in the repository.

---

## 12. Technologies Used

| Technology            | Purpose                                                      |
| --------------------- | ------------------------------------------------------------ |
| Python                | Main programming language                                    |
| FinanceBench          | Financial question-answering benchmark                       |
| PyMuPDF               | Extracting text from PDF documents                           |
| Unstructured          | Extracting structural elements from PDFs                     |
| Gemini 3.5 Flash-Lite | Chunk boundaries for LLM-based chunking; answers and grading |
| Sentence Transformers | Generating embeddings                                        |
| NumPy                 | Vector similarity calculations                               |
| Pandas                | Data processing and result analysis                          |
| ROUGE                 | Measuring text overlap                                       |
| BLEU                  | Measuring text overlap                                       |
| tiktoken              | Token counting and fixed-size chunking                       |

---

## 13. References

**Research Paper:**

Financial Report Chunking for Effective Retrieval Augmented Generation

arXiv:2402.05131

https://arxiv.org/abs/2402.05131

**Dataset:**

FinanceBench

https://github.com/patronus-ai/financebench

---

## 14. Conclusion

This project investigates how different chunking strategies and retrieval designs affect information retrieval from financial reports in RAG systems.

We implemented fixed-size chunking, element-based chunking and LLM-based chunking, and added hierarchical (page-then-chunk) retrieval. On our 42-question evaluation set:

- Standalone element-based and LLM-based chunking did not beat the best fixed-size baselines.
- Hierarchical retrieval consistently improved on its own flat version (no question lost), and combined with LLM chunking it reached 88.1% strict page accuracy, matching Fixed 256 (85.7%) while retrieving about 40% less text.
- End-to-end answer accuracy was statistically indistinguishable across methods (54.8% to 59.5%), so the reduction in retrieved text did not cost answer quality.
- Numerical and table questions remain the main weakness: the right page is often found, but the model answers only 4 to 5 of 11 table-like questions with retrieved text versus 10 of 11 with the gold evidence.

With only 42 questions, one embedding model and one LLM, these results are suggestive rather than conclusive.

**Our main takeaway: better document retrieval is not just about how we divide documents into chunks. It is also about how intelligently we search through those chunks, and for financial tables, finding the right page is not enough.**
