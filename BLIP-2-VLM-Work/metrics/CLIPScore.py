
# pip install git+https://github.com/openai/CLIP.git

import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

GT_PATH = "BLIP_2/data/ground_truth_markets.jsonl"
JSONL_FILE = "BLIP_2/data/blip_model_responses_ZS_markets.jsonl"
IMAGE_DIR = "via-sn-dataset/pakistani_supermarket/"
OUTPUT_FILE = "clipscore_vit_b16_blip2_sm.txt"
# MODEL_NAME = "ViT-L/14"
MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 75  # must be <=77
# ==============================================

print(f"[INFO] Loading CLIP model: {MODEL_NAME}")
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.eval()

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

gt_data = load_jsonl(GT_PATH)
pred_data = load_jsonl(JSONL_FILE)

gt_map = {item["image"]: item["response"] for item in gt_data}

def chunk_text_by_tokens(text, max_tokens=75):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        tokenized = clip.tokenize(" ".join(current_chunk), truncate=False)
        if tokenized.shape[1] > max_tokens:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def compute_clipscore(image_path, caption):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    chunks = chunk_text_by_tokens(caption, MAX_TOKENS)
    sims = []

    for chunk in chunks:
        tokens = clip.tokenize(chunk, truncate=False).to(DEVICE)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T).item()
            sims.append(sim)

    return float(np.mean(sims))

scores = []

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for item in tqdm(pred_data):
        img_name = item["image"]
        model_caption = item["response"]

        if img_name not in gt_map:
            continue

        image_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            continue

        score = compute_clipscore(image_path, model_caption)
        scores.append(score)

        fout.write(f"{img_name}\t{score:.4f}\n")

mean_score = float(np.mean(scores))
print(f"\n[RESULT: BLIP-2 SM] Mean CLIPScore ({MODEL_NAME}): {mean_score:.4f}")

with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
    fout.write(f"\n[RESULT: BLIP-2 SM] MEAN_SCORE\t{mean_score:.4f}\n")
