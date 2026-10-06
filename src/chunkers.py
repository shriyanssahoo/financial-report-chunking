import pymupdf
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def page_texts(pdf_path):
    return [p.get_text() for p in pymupdf.open(pdf_path)]


def fixed_token_chunks(pages, n):
    """Split the whole document into n-token windows, remembering each chunk's pages."""
    tokens, tok_page = [], []
    for i, text in enumerate(pages):
        ids = ENC.encode(text, disallowed_special=())
        tokens.extend(ids)
        tok_page.extend([i] * len(ids))
    chunks = []
    for s in range(0, len(tokens), n):
        chunks.append({
            "text": ENC.decode(tokens[s:s + n]),
            "pages": sorted(set(tok_page[s:s + n])),
        })
    return chunks