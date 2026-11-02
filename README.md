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

All the chunking and retrieval methods were evaluated on the same 42 questions to ensure a consistent comparison.

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
9. Compare the performance of different approaches.

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

Each page of a financial document is represented by a short textual digest.

We take the extracted elements from a page and create a short representation of that page.

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

---

## 6. Experimental Results

All the following results are based on the same 42 questions.

### 6.1 Main Results

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

---

## 7. Key Findings and Observations

Our experiments provided several interesting findings.

### Finding 1: Fixed-Size Chunking Performs Surprisingly Well

We initially expected more sophisticated chunking approaches to outperform fixed-size chunking.

However, the results show that fixed 128-token and 256-token chunking performed better than standalone element-based and LLM-based chunking in terms of strict page accuracy.

This demonstrates that more advanced chunking does not automatically guarantee better retrieval.

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

It did retrieve less text than some fixed-size baselines, but its standalone accuracy remained lower than the best fixed-size approaches.

### Finding 4: Hierarchical Retrieval Improved Performance

This was the most promising result of our experiments.

With P = 10:

- Element-based retrieval improved from 73.8% to 85.7%.
- LLM-based retrieval improved from 78.6% to 88.1%.

It also improved table-question accuracy from 45.5% to 72.7% for both hierarchical configurations.

This suggests that identifying relevant pages before searching individual chunks can be beneficial.

### Finding 5: Better Retrieval Does Not Always Require More Text

Our best-performing configuration, Hierarchical + LLM, achieved 88.1% strict page accuracy while retrieving approximately 5,894 characters on average.

For comparison, Fixed 512 achieved 78.6% accuracy while retrieving approximately 19,370 characters.

This suggests that hierarchical retrieval can achieve better evidence-page retrieval with considerably less retrieved text.

---

## 8. Paired Statistical Analysis

In addition to comparing overall accuracy, we also examined the question-by-question differences between methods.

The purpose was to determine whether hierarchical retrieval consistently helped on individual questions.

The paired comparisons reported in our analysis were:

| Comparison                          | A Wins | B Wins | p-value |
| ----------------------------------- | -----: | -----: | ------: |
| Hierarchical vs Flat (Element)      |      5 |      0 |   0.062 |
| Hierarchical vs Flat (LLM)          |      4 |      0 |   0.125 |
| Hierarchical (Element) vs Fixed 256 |      4 |      4 |   1.000 |
| Hierarchical (LLM) vs Fixed 512     |      6 |      2 |   0.290 |

**Interpretation:**

Hierarchical retrieval consistently improved or maintained performance against its corresponding flat retrieval method in the reported paired comparisons.

However, the p-values are above the conventional 0.05 threshold.

Therefore, these improvements should be described as promising and directionally consistent, rather than statistically significant.

The relatively small evaluation set of 42 questions also limits the strength of statistical conclusions.

---

## 9. Limitations

Although the experiments produced promising results, there are some limitations.

1. **Small Evaluation Dataset:** We evaluated only 42 questions. Larger datasets are required to establish stronger conclusions.

2. **Limited Document Length:** We selected documents containing at most 100 pages. The results may differ for longer financial reports.

3. **Limited Models:** We used one primary embedding model and one LLM for the reported experiments.

4. **Retrieval-Only Evaluation:** We evaluated how effectively the system retrieves evidence. We did not evaluate the correctness of the final answers generated by an LLM.

5. **Element-Based Implementation:** Our implementation follows the rules described in the research paper but uses Unstructured for PDF parsing rather than the paper's Chipper-based approach.

6. **Parameter Selection:** P = 10 performed best, but this value was selected after examining the experimental results. Therefore, the results should be interpreted with this limitation in mind.

---

## 10. Future Improvements

There are several possible directions for extending this project.

These are potential experiments and should not be confused with methods already implemented.

| Proposed Improvement      | Description                                                                                     |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| Contextual Headers        | Add document names and relevant section headings to chunks before embedding.                    |
| Hybrid Search             | Combine keyword-based search (BM25) with semantic embedding search.                             |
| LLM Query Rewriting       | Rewrite questions using relevant financial terminology before retrieval.                        |
| Reranking                 | Retrieve more candidate chunks and use a stronger model to rerank them.                         |
| Page-Vector Pooling       | Create page embeddings by combining the embeddings of individual chunks belonging to that page. |
| Larger Evaluation Dataset | Evaluate the approaches on more FinanceBench questions and longer reports.                      |
| Answer Generation         | Evaluate whether improved retrieval also produces more accurate final answers.                  |

These improvements could help address the ranking problems identified during our experiments, particularly for financial tables and numerical questions.

---

## 11. Repository Structure

The repository is organized into different directories for data, source code and experimental results.

```text
financial-report-chunking/
│
├── data/
│   ├── raw/
│   │   └── Original FinanceBench dataset
│   │
│   └── processed/
│       └── Processed questions and document metadata
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
│   ├── fair_metrics.py
│   ├── diagnose_element.py
│   └── zip_subset.py
│
├── results/
│   └── retrieval/
│       └── Experimental retrieval results and comparison summaries
│
├── subset_pdfs.zip
├── requirements.txt
└── README.md
```

**Note:** Generated intermediate files, PDF parsing outputs, embedding indexes and LLM caches are not all included in the repository.

---

## 12. Technologies Used

The project uses the following technologies and libraries:

| Technology            | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| Python                | Main programming language                           |
| FinanceBench          | Financial question-answering benchmark              |
| PyMuPDF               | Extracting text from PDF documents                  |
| Unstructured          | Extracting structural elements from PDFs            |
| Gemini 3.5 Flash-Lite | Identifying topic boundaries for LLM-based chunking |
| Sentence Transformers | Generating embeddings                               |
| NumPy                 | Vector similarity calculations                      |
| Pandas                | Data processing and result analysis                 |
| ROUGE                 | Measuring text overlap                              |
| BLEU                  | Measuring text overlap                              |
| tiktoken              | Token counting and fixed-size chunking              |

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

This project investigates how different chunking strategies affect information retrieval from financial reports in RAG systems.

We implemented fixed-size chunking, element-based chunking and LLM-based chunking. We then introduced hierarchical retrieval to improve the process of finding relevant information.

Our experiments showed that standalone LLM-based chunking did not outperform the best fixed-size baseline. However, combining LLM-based chunking with hierarchical retrieval achieved our best result of **88.1% strict page accuracy** on the 42-question evaluation set.

The experiments suggest that identifying relevant pages before searching individual chunks can improve retrieval performance while reducing the amount of text retrieved.

The project also highlights the challenges of retrieving numerical information from financial tables and provides several directions for future research.

**Our main takeaway: Better document retrieval is not just about how we divide documents into chunks. It is also about how intelligently we search through those chunks.**
