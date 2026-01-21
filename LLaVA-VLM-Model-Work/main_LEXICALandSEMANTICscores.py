import pandas as pd
from utils.data_loader import load_data
from metrics.bleu import compute_bleu
from metrics.rouge import compute_rouge
from metrics.meteor import compute_meteor
from metrics.cider import compute_cider
from metrics.semantic import compute_bertscore, compute_sbert_similarity
from visualization.eda_plots import plot_radar_chart

# load data files
GT_PATH = 'data/ground_truth_markets.jsonl'
MODEL_PATH = 'data/model_responses_ZS_markets.jsonl'

ids, refs, cands = load_data(GT_PATH, MODEL_PATH)
print(f"Loaded {len(ids)} images for evaluation.")

results_summary = {}

# all computed lexical metrics in metric directory
print("\n computing Lexical metrics")
results_summary.update(compute_bleu(refs, cands))
results_summary.update(compute_rouge(refs, cands))
results_summary.update(compute_meteor(refs, cands))
# cider_agg, _ = compute_cider(ids, refs, cands)
# results_summary.update(cider_agg)

# semantic metrics
print("\n running semantic metrics")
results_summary.update(compute_bertscore(refs, cands))
results_summary.update(compute_sbert_similarity(refs, cands))

# displya final results
print("\n final comprehensive rvaluation result")
df_results = pd.DataFrame([results_summary])
# transpose for easier reading in terminal
print(df_results.T)

# result saving to csv fle
df_results.to_csv('final_semantic_results_summary.csv', index=False)
print("\nResults saved to final_results_summary.csv")

# visualization using plots
print("generating plots")
plot_radar_chart(results_summary)
print("Done....")


# below are only lexical metrics evaluation results
# import pandas as pd
# from utils.data_loader import load_data
# from metrics.bleu import compute_bleu
#
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from metrics.rouge import compute_rouge
# from metrics.meteor import compute_meteor
# from metrics.cider import compute_cider
# from visualization.eda_plots import plot_radar_chart
# from pycocoevalcap.cider.cider import Cider
#
# #  load Data
# GT_PATH = 'data/ground_truth_markets.jsonl'
# MODEL_PATH = 'data/model_responses_ZS_markets.jsonl'
# ids, refs, cands = load_data(GT_PATH, MODEL_PATH)
#
# print(f"Loaded {len(ids)} images for evaluation.")
#
# # compute metrics
# results_summary = {}
#
# print("computing BLEU.")
# bleu_scores = compute_bleu(refs, cands)
# results_summary.update(bleu_scores)
# print("computing ROUGE")
# rouge_scores = compute_rouge(refs, cands)
# results_summary.update(rouge_scores)
# print("computing meteor")
# meteor_scores = compute_meteor(refs, cands)
# results_summary.update(meteor_scores)
# print("computing cider..")
# cider_agg, cider_img_scores = compute_cider(ids, refs, cands)
# results_summary.update(cider_agg)
# # print final results
# print("\nFinal Lexical Evaluation result:")
# df_results = pd.DataFrame([results_summary])
# print(df_results)
# # visualization
# print("generating plots")
# plot_radar_chart(results_summary)
# print("Done.... result summary saved.")
