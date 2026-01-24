import sys
import os
import spacy
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# add project root to path to find utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_loader import load_data

IMAGE_DIR = r"C:\Users\soban\PycharmProjects\LLaVA\via-sn-dataset\pakistani_supermarket"

# load Models
print("Loading NLP and CLIP models (ignore 'use_fast' warning)")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Spacy model not found.")
    sys.exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = SentenceTransformer('clip-ViT-B-32', device=device)


def simplify_caption(text):
    """
    Creates a 'Generic' version of the caption by stripping
    adjectives, adverbs, and spatial details.
    """
    doc = nlp(text)
    # Keep only Nouns, Verbs, and Proper Nouns
    # Remove ADJ (Red), ADV (Quickly), ADP (On/Under - Spatial)
    core_tokens = [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'PROPN', 'NUM']]
    return " ".join(core_tokens)


def calculate_specs(references, candidates, ids):
    print(f"\nComputing SPECS for {len(ids)} images...")

    results = []

    # iterate with progress bar
    for img_id, ref_list, cand in tqdm(zip(ids, references, candidates), total=len(ids)):

        # 1 find image dir
        image_path = os.path.join(IMAGE_DIR, img_id)
        if not os.path.exists(image_path):
            # try appending .jpg if missing
            image_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")

        if not os.path.exists(image_path):
            # skip if image not found
            continue

        try:
            img = Image.open(image_path)

            # 2 prepare texts
            full_text = cand
            generic_text = simplify_caption(cand)

            # 3 compute embeddings
            img_emb = clip_model.encode(img, convert_to_tensor=True)
            full_emb = clip_model.encode(full_text, convert_to_tensor=True)
            generic_emb = clip_model.encode(generic_text, convert_to_tensor=True)

            # 4 compute cosine similarity
            full_score = util.cos_sim(img_emb, full_emb).item()
            generic_score = util.cos_sim(img_emb, generic_emb).item()

            # 5 SPECS = (Specific - Generic) * 100
            # how much "Visual Gain" did the details add?
            specs_score = (full_score - generic_score) * 100

            # store Data
            results.append({
                'image_id': img_id,
                'Full_Caption': full_text,
                'Generic_Caption': generic_text,
                'CLIP_Full': full_score,
                'CLIP_Generic': generic_score,
                'SPECS_Score': specs_score,
                'Word_Count': len(full_text.split())
            })

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    # load Data
    GT_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\ground_truth_markets.jsonl"
    MODEL_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\model_responses_ZS_markets.jsonl"

    print("Loading Dataset")
    ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

    # 2 run calculation
    df_specs = calculate_specs(refs, cands, ids)

    if df_specs.empty:
        print("No results computed. IMAGE_DIR path is incorrect")
    else:
        # 3 show results
        avg_specs = df_specs['SPECS_Score'].mean()
        avg_clip = df_specs['CLIP_Full'].mean()

        print("\n" + "=" * 40)
        print(f"SPECS EVALUATION RESULTS")
        print("=" * 40)
        print(f"Average CLIPScore (Raw):  {avg_clip:.4f}")
        print(f"Average SPECS (Value of Detail): {avg_specs:.4f}")
        print("-" * 40)
        print("INTERPRETATION:")
        print("Positive SPECS: Details (colors, location) matched the image.")
        print("Negative SPECS: Details were hallucinations (conflicted with image).")
        print("=" * 40)

        # 4 save for EDA
        output_file = r"C:\Users\soban\PycharmProjects\LLaVA\analysis\specs_proxy_analysis_results.csv"
        df_specs.to_csv(output_file, index=False)
        
        print(f"\nsaved detailed analysis to '{output_file}'")
        print("use this CSV for the EDA scatter plot in visualization diret:.")
