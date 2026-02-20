"""
Task 2: Build a vector store from a sample of complaints.

Memory-safe version for low-RAM systems.
"""

import argparse
import os
import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=12000)
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/processed/filtered_complaints.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vector_store"
    )
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.random_seed)

    # ---------------------------------------------------------
    # 1. STREAM CSV + SAMPLE FIRST (FIXES OOM)
    # ---------------------------------------------------------
    print("Streaming CSV and sampling...")

    usecols = [
        'mapped_product',
        'cleaned_narrative',
        'Product',
        'Issue',
        'Sub-issue',
        'Company',
        'State',
        'Date received'
    ]

    samples = []
    chunk_size = 50_000

    for chunk in pd.read_csv(
        args.input_file,
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=True
    ):
        chunk = chunk.dropna(subset=['mapped_product', 'cleaned_narrative'])

        for _, row in chunk.iterrows():
            if len(samples) < args.sample_size:
                samples.append(row)
            else:
                # Reservoir sampling
                j = random.randint(0, len(samples))
                if j < args.sample_size:
                    samples[j] = row

        if len(samples) >= args.sample_size:
            break

    df_sample = pd.DataFrame(samples)
    print(f"Sampled {len(df_sample)} complaints.")

    # ---------------------------------------------------------
    # 2. TEXT CHUNKING (UNCHANGED)
    # ---------------------------------------------------------
    print("Chunking narratives...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    chunks = []

    for idx, row in df_sample.iterrows():
        for i, text in enumerate(splitter.split_text(row['cleaned_narrative'])):
            chunks.append({
                "chunk_id": f"{idx}_{i}",
                "text": text,
                "product_category": row['mapped_product'],
                "product": row.get('Product', ''),
                "issue": row.get('Issue', ''),
                "sub_issue": row.get('Sub-issue', ''),
                "company": row.get('Company', ''),
                "state": row.get('State', ''),
                "date_received": row.get('Date received', '')
            })

    chunks_df = pd.DataFrame(chunks)
    print(f"Created {len(chunks_df)} chunks.")

    # ---------------------------------------------------------
    # 3. EMBEDDINGS (BATCHED, SAFE)
    # ---------------------------------------------------------
    print("Embedding in batches...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.IndexFlatL2(384)
    metadata = []

    BATCH = 32
    texts, metas = [], []

    for _, row in chunks_df.iterrows():
        texts.append(row["text"])
        metas.append(row.drop("text").to_dict())

        if len(texts) == BATCH:
            emb = model.encode(texts, show_progress_bar=False)
            index.add(np.asarray(emb, dtype="float32"))
            metadata.extend(metas)
            texts.clear()
            metas.clear()

    if texts:
        emb = model.encode(texts, show_progress_bar=False)
        index.add(np.asarray(emb, dtype="float32"))
        metadata.extend(metas)

    # ---------------------------------------------------------
    # 4. SAVE OUTPUTS
    # ---------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(args.output_dir, "sample_index.faiss"))
    pd.DataFrame(metadata).to_pickle(
        os.path.join(args.output_dir, "sample_chunks.pkl")
    )

    print("Done.")


if __name__ == "__main__":
    main()