import pandas as pd
from utils.data_loader import load_data
from metrics.visual_novel import compute_sliding_clipscore

GT_PATH = 'data/ground_truth_markets.jsonl'
MODEL_PATH = 'data/model_responses_ZS_markets.jsonl'
IMAGE_DIR = 'via-sn-dataset/pakistani_supermarket/'

# load
ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

# run novel metric
agg, scores = compute_sliding_clipscore(IMAGE_DIR, ids, cands)

print("\nNOVELTY ANALYSIS RESULTS ----")
print(f"Standard CLIPScore (Previous): 0.3139")
print(f"Sliding-Window CLIPScore:      {agg['Sliding_CLIPScore']:.4f}")
print("---------------------------")
improvement = ((agg['Sliding_CLIPScore'] - 0.3139) / 0.3139) * 100
print(f"Improvement using novel method: +{improvement:.1f}%")
