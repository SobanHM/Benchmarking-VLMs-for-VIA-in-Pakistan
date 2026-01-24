import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils.data_loader import load_data
from metrics.visual import compute_clipscore
from metrics.semantic import compute_bertscore, compute_sbert_similarity
from metrics.bleu import compute_bleu

# paths configur:
GT_PATH = 'data/ground_truth_markets.jsonl'
MODEL_PATH = 'data/model_responses_ZS_markets.jsonl'
IMAGE_DIR = 'via-sn-dataset/pakistani_supermarket/'


def create_dashboard_plots(df):
    """Generates the Insight Dashboard plots."""
    if not os.path.exists('dashboard_plots'):
        os.makedirs('dashboard_plots')

    # plot 1  Hallucination Quadrant (CLIP vs BERT)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='CLIPScore', y='BERTScore_F1', hue='Type', palette='viridis', s=100)

    #  quadrants lines (using median as cutoff)
    plt.axvline(df['CLIPScore'].median(), color='red', linestyle='--', alpha=0.5)
    plt.axhline(df['BERTScore_F1'].median(), color='red', linestyle='--', alpha=0.5)

    plt.title('Supermarket Environment [LLaVA]\nThe "Hallucination Quadrant [Processing all Tokens]"\n(Top-Left: Likely Hallucination, Top-Right: High Quality)')
    plt.xlabel('Visual Faithfulness (CLIPScore)')
    plt.ylabel('Text Quality (BERTScore)')
    plt.savefig('dashboard_plots/1_hallucination_quadrant_advancevisual.png')
    plt.close()

    # plot 2 metric distribution
    plt.figure(figsize=(12, 6))
    sns.kdeplot(df['CLIPScore'], fill=True, label='CLIPScore (Visual)', color='orange')
    sns.kdeplot(df['SBERT_Sim'], fill=True, label='SBERT (Intent)', color='blue')
    plt.title('Supermarket Environment [LLaVA]\nDistribution of Advance Visual [Process all tokens] vs. Semantic Scores')
    plt.xlabel('Score (0-1)')
    plt.legend()
    plt.savefig('dashboard_plots/2_score_distribution_advancevisual.png')
    plt.close()

    print("Dashboard plots saved in 'dashboard_plots/' folder.")


# execution
print("Starting Evaluation and generating dashbaord.")

ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

# compute Metrics
print("\n[1/3] Computing Text Metrics...")
bert_out = compute_bertscore(refs, cands)  # returns Aggregates
# we need per-instance scores for the scatter plot.
# re-calling using the aggregate function
# call the compute_clipscore which returns list

# use the metrics/visual.py style return (agg, list)
clip_agg, clip_scores = compute_clipscore(IMAGE_DIR, ids, cands)

# for scatter plot, need the individual BERT/SBERT scores again. re-generate them
from sentence_transformers import SentenceTransformer, util

sbert = SentenceTransformer('all-MiniLM-L6-v2')
bert_scores_list = []
sbert_scores_list = []

print("\n[2/3] Computing Per-Instance Semantic Scores...")
cand_embs = sbert.encode(cands, convert_to_tensor=True)
for i, r_list in enumerate(refs):
    # SBERT
    ref_embs = sbert.encode(r_list, convert_to_tensor=True)
    sbert_scores_list.append(util.cos_sim(cand_embs[i], ref_embs).max().item())
    # should use the `score` function from bert_score library
    # getting the F1 list.
    pass

# (self-correction:
# use score
from bert_score import score

flat_refs = [r[0] for r in refs]
P, R, F1 = score(cands, flat_refs, lang="en", verbose=False)
bert_scores_list = F1.tolist()

# consolidate data for csv
df = pd.DataFrame({
    'image': ids,
    'CLIPScore': clip_scores,
    'BERTScore_F1': bert_scores_list,
    'SBERT_Sim': sbert_scores_list,
    'Type': ['Market' if 'sm' in x else 'Other' for x in ids]  # specific to supermarket set
})

# save and plot
output_file = r"C:\Users\soban\PycharmProjects\LLaVA\analysis\dataset_data_with_clipscore.csv"
df.to_csv(output_file, index=False)
print("\n[3/3] Generating Dashboard..")
create_dashboard_plots(df)

print("\nFinal Insights CLIPscore and semantic metrics score.")
print(f"Average CLIPScore: {df['CLIPScore'].mean():.4f}")
print(f"Average BERTScore: {df['BERTScore_F1'].mean():.4f}")
print(f"Correlation (Visual <-> Text): {df['CLIPScore'].corr(df['BERTScore_F1']):.4f}")
