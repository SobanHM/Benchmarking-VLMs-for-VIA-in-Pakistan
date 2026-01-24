import pandas as pd
import numpy as np
import torch
from bert_score import score
from sentence_transformers import SentenceTransformer, util
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_data
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# setup on gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def get_detailed_metrics(ids, references, candidates):
    data = []

    print("Computing per-instance scores...")

    # pre-compute BERTScore use first reference for BERTScore 1:1 comparison
    flat_refs = [r[0] for r in references]
    P, R, F1 = score(candidates, flat_refs, lang="en", verbose=True, device=device)

    # Pre-compute SBERT Embeddings
    cand_embs = sbert_model.encode(candidates, convert_to_tensor=True)

    chencherry = SmoothingFunction()

    for i, img_id in enumerate(ids):
        # bleu 4 : lexical
        ref_tokens = [r.lower().split() for r in references[i]]
        cand_tokens = candidates[i].lower().split()
        bleu4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

        # bertscore : semantic tokens
        b_score = F1[i].item()

        # sentence bert as semantic global
        # compare candidate to refs for this image and take max
        ref_embs = sbert_model.encode(references[i], convert_to_tensor=True)
        cos_scores = util.cos_sim(cand_embs[i], ref_embs)
        sbert_val = torch.max(cos_scores).item()

        # store data for per image analysis
        data.append({
            'image': img_id,
            'Ground_Truth': references[i][0],
            'Model_Response': candidates[i],
            'BLEU_4': round(bleu4, 4),
            'BERTScore_F1': round(b_score, 4),
            'SBERT_Sim': round(sbert_val, 4),
            'Length_Diff': len(candidates[i].split()) - len(references[i][0].split())
        })

    return pd.DataFrame(data)

# paths and executipn
GT_PATH = '../data/ground_truth_markets.jsonl'
MODEL_PATH = '../data/model_responses_ZS_markets.jsonl'

print("Loading data...")
ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

df = get_detailed_metrics(ids, refs, cands)

# saving in csv fiel for manual inspection
output_file = 'evaluation_metrics_scores_analysis_of_sbert.csv'
df.to_csv(output_file, index=False)
print(f"Done.. Analysis saved to {output_file}")
