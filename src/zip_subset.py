# src/zip_subset.py
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
docs = pd.read_csv(ROOT / "data/processed/subset_docs.csv")
with zipfile.ZipFile(ROOT / "subset_pdfs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for d in docs["doc_name"]:
        z.write(ROOT / "data/pdfs" / f"{d}.pdf", f"{d}.pdf")
print("Wrote subset_pdfs.zip")