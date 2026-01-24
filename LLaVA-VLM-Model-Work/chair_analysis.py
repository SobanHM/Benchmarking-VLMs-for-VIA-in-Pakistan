import pandas as pd
from utils.data_loader import load_data
from metrics.hallucination import compute_chair
from metrics.smart_chair import compute_smart_chair
from metrics.semantic_chair import  compute_semantic_chair

GT_PATH = 'data/ground_truth_markets.jsonl'
MODEL_PATH = 'data/model_responses_ZS_markets.jsonl'

# loading cnfgs
ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

# run CHAIR
metrics, details = compute_chair(refs, cands)

print("\n Hallucination Analysis (CHAIR) ----")
print(f"CHAIR-i (Object Hallucination %):   {metrics['CHAIR-i (Object Error Rate)'] * 100:.2f}%")
print(f"CHAIR-s (Sentences with Lies %):    {metrics['CHAIR-s (Sentence Error Rate)'] * 100:.2f}%")
print("--------------------------------------")
print("Interpretation:")
print("- CHAIR-i: Percentage of mentioned objects that are NOT in the ground truth.")
print("- CHAIR-s: Percentage of descriptions that contain at least one hallucination.")

# show a failure case
df_details = pd.DataFrame(details)
df_details['Model_Response'] = cands
df_details['Ground_Truth'] = [r[0] for r in refs]

# hallucinations results in objects and sentence interpretation
failures = df_details[df_details['CHAIR_Score'] == 1]
if not failures.empty:
    sample = failures.iloc[0]
    print("\n[Sample Hallucination Failure]")
    print(f"Model Said:.. {sample['Model_Response'][:100]} ")
    print(f"Hallucinated Objects Detected: {sample['Hallucinated_Objects']}")


GT_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\ground_truth_markets.jsonl"
MODEL_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\model_responses_ZS_markets.jsonl"

ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

ci, cs = compute_smart_chair(refs, cands)

print("\nSmartly handled CHAIR RESULTS (with some supermarket vocab) -----")
print(f"Refined CHAIR-i (Object Error):   {ci * 100:.2f}%")
print(f"Refined CHAIR-s (Sentence Error): {cs * 100:.2f}%")

print("\n\n Semantically improved CHAIR RESULTS ---")
ids, refs, cands = load_data(GT_PATH, MODEL_PATH)
ci, cs = compute_semantic_chair(refs, cands)

print("\n SEMANTIC RESULTS (SBERT Matching)------")
print(f"Semantic CHAIR-i: {ci * 100:.2f}%")
print(f"Semantic CHAIR-s: {cs * 100:.2f}%")
