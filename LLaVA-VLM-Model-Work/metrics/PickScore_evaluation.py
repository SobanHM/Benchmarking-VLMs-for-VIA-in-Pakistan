import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# configs
JSONL_FILE = "data/blip_model_responses_ZS_markets.jsonl"
IMAGE_DIR = "via-sn-dataset/pakistani_supermarket/"
OUTPUT_CSV = "pickscore_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# official PickScore model
PICKSCORE_MODEL = "yuvalkirstain/pickscore_v1"
CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

print("[INFO] Loading PickScore model..")
processor = AutoProcessor.from_pretrained(CLIP_MODEL)
model = AutoModel.from_pretrained(PICKSCORE_MODEL).to(DEVICE)
model.eval()

results = []

def compute_pickscore(image_path, caption):
    """
    computes PickScore(image, caption)
    Uses official PickScore forward pass.
    Long captions are handled safely (auto-truncated by tokenizer).
    """

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=caption,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits_per_image.squeeze().item()

    return float(score)

print("[INFO] Starting PickScore evaluation..")

with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        data = json.loads(line)

        image_file = data["image"]
        caption = data["response"]

        image_path = os.path.join(IMAGE_DIR, image_file)

        if not os.path.exists(image_path):
            print(f"[WARNING] Image not found: {image_path}")
            continue

        score = compute_pickscore(image_path, caption)

        results.append({
            "image": image_file,
            "pickscore": score
        })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n[INFO] Evaluation completed!")
print(f"[INFO] Results saved to: {OUTPUT_CSV}")
print(f"[RESULT] Mean PickScore of LLaVA sm: {df['pickscore'].mean():.4f}")
