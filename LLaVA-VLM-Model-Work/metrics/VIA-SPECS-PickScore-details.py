import os
import sys
import torch
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import spacy

# --- Research & Model Imports ---
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer, util

# --- Project Imports ---
# Assumes your project structure has 'utils' folder in root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

try:
    from utils.text_filters import (
        spatial_filter, object_filter, hazard_filter,
        action_filter, context_filter, dimension_generic_filter
    )
except ImportError:
    print("CRITICAL ERROR: Could not import text filters from 'utils.text_filters'.")
    print("Ensure you are running this from the correct directory.")
    sys.exit(1)

IMAGE_DIR = r"C:\Users\soban\PycharmProjects\LLaVA\via-sn-dataset\pakistani_supermarket"
MODEL_RESPONSES_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\model_responses_ZS_markets.jsonl"
OUTPUT_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\analysis\advanced_metrics_results_pickandspecs-via.csv"

# Set Device (GPU is highly recommended for PickScore)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Running Evaluation on: {DEVICE} ---")

# --- Metric 1: PickScore Setup ---
# Why: PickScore acts as a proxy for "Human Preference".
# It tells us if the caption feels natural and helpful to a human user.
print("Loading PickScore Model (ViT-H-14 backbone)...")
pick_processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(DEVICE)

# --- Metric 2: VIA-SPECS Setup ---
# Why: SPECS measures "Information Gain".
# It proves that the added details (verbosity) actually align with the image.
print("Loading CLIP Model for SPECS (ViT-B-32)...")
specs_model = SentenceTransformer("clip-ViT-B-32", device=DEVICE)
nlp = spacy.load("en_core_web_sm")


def get_pick_score(image, text):
    """
    Calculates the raw probability score that a human would 'pick' this text for this image.
    """
    # Preprocess inputs
    image_inputs = pick_processor(images=image, padding=True, truncation=True, max_length=77, return_tensors="pt").to(
        DEVICE)
    text_inputs = pick_processor(text=text, padding=True, truncation=True, max_length=77, return_tensors="pt").to(
        DEVICE)

    with torch.no_grad():
        # Embed
        image_emb = pick_model.get_image_features(**image_inputs)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        text_emb = pick_model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Score: Dot product implies alignment likelihood
        score = (text_emb @ image_emb.T).item()

    return score


def get_specs_score(image, text):
    """
    Calculates VIA-SPECS score by comparing Specific vs. Generic alignment.
    Returns the composite score and the breakdown.
    """
    # Dimensions for Navigation (The 5 Blocks)
    dimensions = {
        "spatial": (spatial_filter, 0.30),
        "hazard": (hazard_filter, 0.25),
        "action": (action_filter, 0.20),
        "object": (object_filter, 0.15),
        "context": (context_filter, 0.10)
    }

    dim_scores = {}
    weighted_score = 0.0

    # Encode Image Once
    img_emb = specs_model.encode(image, convert_to_tensor=True)

    for name, (filter_fn, weight) in dimensions.items():
        # 1. Extract Specific Details
        specific_text = filter_fn(text, nlp)

        # 2. Create Generic Baseline (The "Control Group")
        generic_text = dimension_generic_filter(specific_text, nlp, name)

        # 3. Compare Alignment (A/B Test)
        spec_emb = specs_model.encode(specific_text, convert_to_tensor=True)
        gen_emb = specs_model.encode(generic_text, convert_to_tensor=True)

        spec_sim = util.cos_sim(img_emb, spec_emb).item()
        gen_sim = util.cos_sim(img_emb, gen_emb).item()

        # 4. Calculate Gain (Did the detail help?)
        gain = (spec_sim - gen_sim) * 100
        dim_scores[f"{name}_gain"] = gain
        weighted_score += gain * weight

    return weighted_score, dim_scores


def load_jsonl_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    # Load Model Responses
    print(f"Loading responses from: {MODEL_RESPONSES_PATH}")
    model_data = load_jsonl_data(MODEL_RESPONSES_PATH)

    results = []

    print("Starting Advanced Evaluation Loop...")
    # Using enumerate to track line numbers for debugging
    for i, entry in enumerate(tqdm(model_data)):

        # 1. ROBUST ID EXTRACTION
        # Try all common variations of ID keys
        img_id = entry.get('image') or entry.get('id') or entry.get('question_id')

        # SAFETY CHECK: If we still have None, skip this line and warn the user
        if img_id is None:
            print(f"\n[WARNING] Skipping Line {i + 1}: No 'image_id' found in data: {entry}")
            continue

        # Ensure img_id is a string (sometimes IDs are loaded as integers)
        img_id = str(img_id)

        # 2. ROBUST TEXT EXTRACTION
        # Try all common variations of text keys
        response_text = entry.get('response') or entry.get('description') or entry.get('caption')

        if not response_text:
            print(f"\n[WARNING] Skipping Line {i + 1} (ID: {img_id}): No response text found.")
            continue

        # Path Handling
        img_filename = img_id if img_id.endswith('.jpg') else f"{img_id}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)

        # Skip if image missing
        if not os.path.exists(img_path):
            # Optional: Print only if you suspect path issues
            # print(f"Image not found: {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening {img_filename}: {e}")
            continue

        # --- Calculation Block ---

        # 1. Calculate PickScore (Quality)
        pick_score = get_pick_score(image, response_text)

        # 2. Calculate SPECS (Utility)
        specs_final, specs_breakdown = get_specs_score(image, response_text)

        # Store Result
        row = {
            "image_id": img_id,
            "PickScore": pick_score,
            "VIA_SPECS": specs_final,
            # Flatten SPECS details
            **specs_breakdown
        }
        results.append(row)

    # Save to CSV
    if not results:
        print("CRITICAL: No results generated. Check your paths and JSON keys.")
        return

    df = pd.DataFrame(results)

    # Calculate Averages
    print("\n" + "=" * 40)
    print("FINAL RESULTS SUMMARY")
    print("=" * 40)
    print(f"Total Images Evaluated: {len(df)}")
    print(f"Avg PickScore (Fluency): {df['PickScore'].mean():.4f}")
    print(f"Avg VIA-SPECS (Utility): {df['VIA_SPECS'].mean():.4f}")
    print("-" * 20)
    # Check if we actually have gain columns before printing (in case of empty results)
    if 'spatial_gain' in df.columns:
        print("SPECS Breakdown (Avg Gain):")
        print(f"Spatial: {df['spatial_gain'].mean():.2f}")
        print(f"Hazard:  {df['hazard_gain'].mean():.2f}")
        print(f"Action:  {df['action_gain'].mean():.2f}")
    print("=" * 40)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDetailed analysis saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
