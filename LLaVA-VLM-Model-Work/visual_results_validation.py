import pandas as pd
import torch
import os
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from utils.data_loader import load_data


GT_PATH = 'data/ground_truth_markets.jsonl'
MODEL_PATH = 'data/model_responses_ZS_markets.jsonl'
IMAGE_DIR = 'via-sn-dataset/pakistani_supermarket/'

def validate_clip_performance():
    print("Starting CLIP Validation Experiment---")

    # 1 loading data
    ids, refs, cands = load_data(GT_PATH, MODEL_PATH)
    # flatten refs (take one human caption per image)
    human_texts = [r[0] for r in refs]

    # 2 load Model on gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}")
    model = SentenceTransformer('clip-ViT-B-32', device=device)

    # 3 compute scores
    print("Encoding Images..")
    img_embs = []
    valid_ids = []

    for i, img_id in enumerate(ids):
        path = os.path.join(IMAGE_DIR, img_id)
        try:
            img = Image.open(path)
            img_embs.append(model.encode(img, convert_to_tensor=True))
            valid_ids.append(i)
        except:
            print(f"Skipping missing image: {img_id}")

    if not valid_ids:
        print("Error: No images found. Check your path.")
        return

    img_embs = torch.stack(img_embs)

    # filter text to match valid images
    valid_human_texts = [human_texts[i] for i in valid_ids]
    valid_model_texts = [cands[i] for i in valid_ids]

    print("Encoding Text..")
    human_emb = model.encode(valid_human_texts, convert_to_tensor=True)
    model_emb = model.encode(valid_model_texts, convert_to_tensor=True)

    # 4 calculate similarity
    human_scores = util.cos_sim(img_embs, human_emb).diagonal()
    model_scores = util.cos_sim(img_embs, model_emb).diagonal()

    # 5 report
    avg_human = human_scores.mean().item()
    avg_model = model_scores.mean().item()

    print("\n Validation Results ======")
    print(f"Human Ground Truth CLIPScore: {avg_human:.4f}")
    print(f"Your Model CLIPScore:         {avg_model:.4f}")
    print(f"------------")
    print(f"Model Performance vs Human:   {(avg_model / avg_human) * 100:.1f}%")


    if avg_human < 0.40:
        print("\nINSIGHT: The Human score is also low (< 0.40).")
        print("This confirms that '0.31' is NOT a failure.")
        print("It is simply the scale of CLIPScore for this specific dataset.")


if __name__ == "__main__":
    validate_clip_performance()

"""
    Results:
        Encoding Images...
        Encoding Text...

        Validation Results ======
        Human Ground Truth CLIPScore: 0.3271
        Your Model CLIPScore:         0.3139
        --------------------------------
        Model Performance vs Human:   96.0%

        INSIGHT: The Human score is also low (< 0.40).
        This confirms that '0.31' is NOT a failure.
"""
