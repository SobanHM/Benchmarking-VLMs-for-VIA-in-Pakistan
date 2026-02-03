
import os
import sys
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor, AutoModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.data_loader import load_data
from utils.text_filters import (
    spatial_filter,
    object_filter,
    hazard_filter,
    action_filter,
    context_filter,
    dimension_generic_filter
)

GT_PATH = "data/ground_truth_markets.jsonl"
# MODEL_PATH = "data/blip_results_supermarket.jsonl"
JSONL_FILE = "data/model_responses_ZS_markets.jsonl"
IMAGE_DIR = "via-sn-dataset/pakistani_supermarket/"
OUTPUT_DIR = "analysis/results_llava_sm_with_VIApickscorespecs.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] Loading NLP + CLIP + PickScore models...")
nlp = spacy.load("en_core_web_sm")
clip_model = SentenceTransformer(
    "clip-ViT-B-32",
    device=DEVICE
)

PICKSCORE_MODEL = "yuvalkirstain/pickscore_v1"
CLIP_BACKBONE = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

pick_processor = AutoProcessor.from_pretrained(CLIP_BACKBONE)
pick_model = AutoModel.from_pretrained(PICKSCORE_MODEL).to(DEVICE)
pick_model.eval()

def clip_score(image, text):
    if not text.strip():
        return 0.0
    img_emb = clip_model.encode(image, convert_to_tensor=True)
    txt_emb = clip_model.encode(text, convert_to_tensor=True)
    return util.cos_sim(img_emb, txt_emb).item()


def pickscore(image, caption):
    if not caption.strip():
        return 0.0

    inputs = pick_processor(
        images=image,
        text=caption,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = pick_model(**inputs)
        return outputs.logits_per_image.squeeze().item()


def compute_via_specs_with_pickscore():

    ids, refs, cands = load_data(GT_PATH, MODEL_PATH)
    results = []

    dimensions = {
        "spatial": spatial_filter,
        "object": object_filter,
        "hazard": hazard_filter,
        "action": action_filter,
        "context": context_filter
    }

    for img_id, gt, cand in tqdm(zip(ids, refs, cands), total=len(ids)):

        img_path = os.path.join(IMAGE_DIR, img_id)
        if not img_path.lower().endswith(".jpg"):
            img_path += ".jpg"

        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")

        dim_scores = {}
        pick_deltas = {}

        for name, fn in dimensions.items():

            full = fn(cand, nlp)
            generic = dimension_generic_filter(full, nlp, name)

            # CLIP-SPECS
            clip_full = clip_score(image, full)
            clip_generic = clip_score(image, generic)
            dim_scores[name] = (clip_full - clip_generic) * 100

            # PickScore Δ
            pick_full = pickscore(image, full)
            pick_generic = pickscore(image, generic)
            pick_deltas[name] = pick_full - pick_generic

        # VIA-SPECS (CLIP-based)
        via_specs = (
            0.30 * dim_scores["spatial"] +
            0.25 * dim_scores["hazard"] +
            0.20 * dim_scores["action"] +
            0.15 * dim_scores["object"] +
            0.10 * dim_scores["context"]
        )

        # VIA-PickScore (human-aligned)
        via_pickscore = sum(pick_deltas.values()) / len(pick_deltas)

        results.append({
            "image_id": img_id,
            **{f"{k}_SPECS": v for k, v in dim_scores.items()},
            **{f"{k}_PickDelta": v for k, v in pick_deltas.items()},
            "VIA_SPECS": via_specs,
            "VIA_PickScore": via_pickscore
        })

    df = pd.DataFrame(results)

    output_file = "analysis/via_specs_pickscore_results_llava_sm.csv"
    df.to_csv(output_file, index=False)

    print("\n[INFO] Evaluation complete")
    print(f"[INFO] Saved: {output_file}")
    print(f"[STATS] Mean VIA-SPECS: {df['VIA_SPECS'].mean():.2f}")
    print(f"[STATS] Mean VIA-PickScore Δ: {df['VIA_PickScore'].mean():.4f}")


if __name__ == "__main__":
    compute_via_specs_with_pickscore()