import sys
import os
import torch
import pandas as pd
import nltk
from sentence_transformers import CrossEncoder

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_loader import load_data

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


print("Loading AI Judge (Context-Aware NLI)..")
device = "cuda" if torch.cuda.is_available() else "cpu"
# This model is trained to determine: "Does Sentence A imply Sentence B?"
model = CrossEncoder('cross-encoder/nli-distilroberta-base', device=device)

# safety hypothesis (what we are looking for)
# model checks if the text entails any of these.
SAFETY_CRITERIA = {
    "Entrances_Exits": [
        "There is a door, gate, or entrance.",
        "There is a way out or exit.",
        "There is a glass frontage leading outside.",
        "There are sliding doors."
    ],
    "Level_Changes": [
        "There are stairs or steps.",
        "There is an elevator or escalator.",
        "There is a ramp."
    ],
    "Trip_Hazards": [
        "There are obstacles on the floor like boxes or luggage.",
        "There are wires, cables, or trash on the ground.",
        "The floor is wet or uneven."
    ],
    "Structural_Obstacles": [
        "There is a pillar, column, or pole.",
        "There is a wall or barrier.",
        "There is a fence."
    ],
    "Dynamic_Obstacles": [
        "This text describes people, person, customers, or a crowd nearby.",
        "This text mentions a person walking or standing in the area.",
        "This text describes dynamic movement of people."
    ]
}


# 3 logic: cumulative context builder
def check_concept_presence(text, hypotheses):
    """
    Implements the 'Context Building' logic.
    Instead of checking single sentences we check large chunks of context.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences: return False, 0.0, ""

    # strategy: create large overlapping windows to capture context
    # window size 5 ensures we capture "Context + Subject + Action"
    WINDOW_SIZE = 5
    STRIDE = 2

    context_windows = []

    # if text is short, take it all
    if len(sentences) <= WINDOW_SIZE:
        context_windows.append(" ".join(sentences))
    else:
        # Sliding Window
        for i in range(0, len(sentences), STRIDE):
            window = " ".join(sentences[i: i + WINDOW_SIZE])
            context_windows.append(window)
            if i + WINDOW_SIZE >= len(sentences):
                break

    # prepare nli pairs: (Context_Window, Hypothesis)
    pairs = []
    for window in context_windows:
        for hyp in hypotheses:
            pairs.append([window, hyp])

    if not pairs: return False, 0.0, ""

    # judgement step
    scores = model.predict(pairs)

    # scores: [prob_contradiction, prob_entailment, prob_neutral] (depends on model)
    # cross-encoder/nli-distilroberta-base outputs: [contradiction, entailment, neutral] ??
    # actually this model outputs logits for [Contradiction, Entailment, Neutral] usually
    # OR [Contradiction, Entailment]. Lets check docs or use index 1 (Entailment).
    # For 'cross-encoder/nli-distilroberta-base', label mapping is:
    # 0: Contradiction, 1: Entailment, 2: Neutral (Wait, standard is usually Entailment=1 or 2)
    # Lets assume standard MNLI: 0=Contradiction, 1=Entailment (if 2 classes) or similar.
    # We will look for the HIGHEST score in the 'Entailment' column.
    # we use a softmax to get probabilities if needed, but raw logits work for max.
    # trust the logic: Index 1 is usually Entailment in 2-class, or Index 2 in 3-class?
    # Actually, let's use a simpler heuristic:
    # If using 'cross-encoder/nli-distilroberta-base', it is trained on SNLI/MNLI.
    # Labels: 0: contradiction, 1: entailment, 2: neutral.

    entailment_scores = scores[:, 1]  # Index 1 = Entailment

    max_score = entailment_scores.max()
    best_idx = entailment_scores.argmax()
    best_window = pairs[best_idx][0]

    # threshold: 0.0 is usually the logit midpoint.
    # > 1.0 indicates strong confidence and > 2.0 is very strong.
    IS_PRESENT = max_score > 0.5
    return IS_PRESENT, max_score, best_window


# 4 audit exec: with debugging
def run_context_audit(references, candidates, ids):
    print(f"Running Safety Audit on {len(ids)} samples...")

    stats = {cat: {'TP': 0, 'FP': 0, 'FN': 0} for cat in SAFETY_CRITERIA}

    # debugging complex images :my doubt
    DEBUG_LIST = ['sm_1', 'sm_8','sm_20', 'sm_25', 'sm_76']

    for img_id, ref_list, cand in zip(ids, references, candidates):

        # 1 analyze GT
        gt_result = {}
        for cat, hyps in SAFETY_CRITERIA.items():
            found = False
            evidence = ""
            for ref in ref_list:
                present, score, ev = check_concept_presence(ref, hyps)
                if present:
                    found = True
                    evidence = ev
                    break
            gt_result[cat] = {'present': found, 'ev': evidence}

        # 2 analyze model
        model_result = {}
        for cat, hyps in SAFETY_CRITERIA.items():
            present, score, ev = check_concept_presence(cand, hyps)
            model_result[cat] = {'present': present, 'ev': ev}

        # 3 compare & score
        for cat in SAFETY_CRITERIA:
            in_gt = gt_result[cat]['present']
            in_model = model_result[cat]['present']

            if in_gt and in_model:
                stats[cat]['TP'] += 1
            elif in_model and not in_gt:
                stats[cat]['FP'] += 1
            elif in_gt and not in_model:
                stats[cat]['FN'] += 1

            # 4 printing debugged images info
            if any(d in img_id for d in DEBUG_LIST) and cat == "Entrances_Exits":
                print(f"\nContext check: {img_id} [{cat}]")
                print(f"  GT Entails?    {in_gt}  (Context: \"{gt_result[cat]['ev'][:80]}...\")")
                print(f"  Model Entails? {in_model}  (Context: \"{model_result[cat]['ev'][:80]}...\")")
                if in_gt and not in_model:
                    print("  Verdict: OMISSION (Model failed context check)")
                elif not in_gt and in_model:
                    print("  Verdict: HALLUCINATION")
                else:
                    print("  Verdict: MATCH")

    # 5 final stats
    data = []
    for cat, s in stats.items():
        tp, fp, fn = s['TP'], s['FP'], s['FN']

        # safety Metrics
        model_claims = tp + fp
        real_hazards = tp + fn

        hallucination_rate = (fp / model_claims * 100) if model_claims > 0 else 0.0
        omission_rate = (fn / real_hazards * 100) if real_hazards > 0 else 0.0

        data.append({
            'Category': cat,
            'Hallucination_Rate': hallucination_rate,
            'Omission_Rate': omission_rate
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    GT_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\ground_truth_markets.jsonl"
    MODEL_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\model_responses_ZS_markets.jsonl"

    ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

    df_results = run_context_audit(refs, cands, ids)

    print("\n" + "=" * 80)
    print(f"{'CATEGORY':<25} | {'FALSE ALARM':<15} | {'MISSED HAZARD'}")
    print("=" * 80)
    for _, row in df_results.iterrows():
        print(f"{row['Category']:<25} | {row['Hallucination_Rate']:6.1f}%{' ' * 8} | {row['Omission_Rate']:6.1f}%")
    print("=" * 80)
    output_file = r"C:\Users\soban\PycharmProjects\LLaVA\analysis\safety_context_scores.csv"
    df_results.to_csv(output_file, index=False)
