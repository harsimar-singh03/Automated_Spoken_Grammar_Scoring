"""
Extract Top-4 Layer Pooled Embeddings from WavLM Base
Layers 9, 10, 11, 12 capture both phonetic/acoustic grammar transitions and high-level prosody.
Uses batch size 8 on CPU with torch.set_num_threads(12).
Saves:
- D:\\shl\\working\\train_wavlm_top4.npy (769, 768)
- D:\\shl\\working\\test_wavlm_top4.npy (216, 768)
"""
import os, sys, time, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, torch
from transformers import WavLMModel
from scipy.io import wavfile
from tqdm import tqdm

warnings.filterwarnings('ignore')
torch.set_num_threads(12)

DATA_DIR   = r"D:\SHL\shl-hiring-assessment-2026\Dataset_Final"
OUTPUT_DIR = r"D:\shl\working"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Loading microsoft/wavlm-base model...")
model = WavLMModel.from_pretrained('microsoft/wavlm-base')
model.eval()
print("Model loaded successfully.")

def process_file(path, target_sr=16000, max_dur=30):
    try:
        sr, sig = wavfile.read(path)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        if sig.dtype != np.float32:
            sig = sig.astype(np.float32) / 32768.0
        max_samples = target_sr * max_dur
        if len(sig) > max_samples:
            sig = sig[:max_samples]
        elif len(sig) < max_samples:
            sig = np.pad(sig, (0, max_samples - len(sig)), mode='constant')
        return sig
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros(target_sr * max_dur, dtype=np.float32)

def extract_dataset(df, folder_name, out_path, batch_size=8):
    if os.path.exists(out_path):
        print(f"File {out_path} already exists! Loading directly.")
        return np.load(out_path)
    
    n_total = len(df)
    embeddings = []
    print(f"\nExtracting {n_total} files from {folder_name} (batch_size={batch_size})...")
    t0 = time.time()
    
    for start_idx in range(0, n_total, batch_size):
        end_idx = min(start_idx + batch_size, n_total)
        batch_files = [os.path.join(DATA_DIR, folder_name, fn) for fn in df["filename"].iloc[start_idx:end_idx]]
        batch_sigs = [process_file(fp) for fp in batch_files]
        
        batch_tensor = torch.tensor(np.array(batch_sigs), dtype=torch.float32)
        with torch.no_grad():
            out = model(batch_tensor, output_hidden_states=True)
            # Top 4 hidden states: hs[-4:] -> stack to (4, B, time, 768)
            top4 = torch.stack(out.hidden_states[-4:], dim=0).mean(dim=0) # (B, time, 768)
            pooled = top4.mean(dim=1).numpy() # (B, 768)
            embeddings.append(pooled)
        
        elapsed = time.time() - t0
        processed = end_idx
        rate = processed / elapsed
        rem_sec = (n_total - processed) / max(rate, 1e-5)
        print(f"  [{folder_name}] {processed:3d}/{n_total} files | Speed: {rate:.2f} files/s | ETA: {rem_sec/60:.1f} min", flush=True)
    
    all_emb = np.vstack(embeddings)
    np.save(out_path, all_emb)
    print(f"Successfully saved: {out_path} | Shape: {all_emb.shape} | NaNs: {np.isnan(all_emb).sum()}")
    return all_emb

# 1. Extract Train
train_out = os.path.join(OUTPUT_DIR, "train_wavlm_top4.npy")
extract_dataset(train_df, "train", train_out, batch_size=8)

# 2. Extract Test
test_out = os.path.join(OUTPUT_DIR, "test_wavlm_top4.npy")
extract_dataset(test_df, "test", test_out, batch_size=8)

print("\nWavLM Top-4 Extraction Complete!")
