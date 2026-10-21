"""
Extract ELECTRA-base-discriminator embeddings for train and test transcripts.
ELECTRA is pre-trained with replaced token detection (RTD), making it an expert
in grammar, unnatural phrasing, and syntactic anomalies.
"""
import os, sys, time, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings('ignore')

DATA_DIR   = r"D:\SHL\shl-hiring-assessment-2026\Dataset_Final"
OUTPUT_DIR = r"D:\shl\working"
TRANS_DIR  = r"D:\shl\working\transcripts"

train_df    = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df     = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_trans = pd.read_csv(os.path.join(TRANS_DIR, "train_transcripts.csv"))
test_trans  = pd.read_csv(os.path.join(TRANS_DIR, "test_transcripts.csv"))

train_df_m = train_df.merge(train_trans, on="filename", how="left")
test_df_m  = test_df.merge(test_trans,  on="filename", how="left")
train_texts = train_df_m["transcript"].fillna("").astype(str).tolist()
test_texts  = test_df_m["transcript"].fillna("").astype(str).tolist()

model_name = "google/electra-base-discriminator"
print(f"Loading {model_name}...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s")

def extract(texts, desc="Extracting"):
    embs = []
    t_start = time.time()
    for text in tqdm(texts, desc=desc):
        text_str = str(text or "").strip()
        inputs = tok(text_str, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
            # Mean pooling across tokens
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).numpy()
        embs.append(emb)
    print(f"Completed {len(texts)} in {time.time()-t_start:.1f}s")
    return np.array(embs, dtype=np.float32)

print("\nExtracting train transcripts (769 samples)...")
train_embs = extract(train_texts, "Train ELECTRA")
train_out = os.path.join(OUTPUT_DIR, "train_electra_embeddings.npy")
np.save(train_out, train_embs)
print(f"Saved: {train_out} | Shape: {train_embs.shape} | NaNs: {np.isnan(train_embs).sum()}")

print("\nExtracting test transcripts (216 samples)...")
test_embs = extract(test_texts, "Test ELECTRA")
test_out = os.path.join(OUTPUT_DIR, "test_electra_embeddings.npy")
np.save(test_out, test_embs)
print(f"Saved: {test_out} | Shape: {test_embs.shape} | NaNs: {np.isnan(test_embs).sum()}")

print("\nELECTRA feature extraction complete!")
