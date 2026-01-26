import sys
import os
import json
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

# setup paths to avoid directory/file access issues, python a root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_loader import load_data

# initialize NLP tools
lemmatizer = WordNetLemmatizer()

# 1 defineing taxonomy of failure : logical bucket
TAXONOMY = {
    "Vertical Navigation (CRITICAL)": {
        "stairs", "staircase", "steps", "ladder", "ramp", "elevator", "escalator"
    },
    "Exits & Entrances (CRITICAL)": {
        "door", "doorway", "gate", "entrance", "exit", "opening"
    },
    "Pathways (Safety)": {
        "path", "walkway", "aisle", "corridor", "hallway", "floor", "ground", "pavement"
    },
    "Obstacles (Hazards)": {
        "barrier", "post", "pole", "column", "pillar", "stand", "stall", "counter", "table", "chair", "box", "crate"
    },
    "Spatial Relations": {
        "left", "right", "center", "middle", "background", "foreground", "top", "bottom", "side"
    },
    "Crowd/People (Transient)": {
        "people", "person", "man", "woman", "crowd", "customer", "shopper", "vendor", "seller"
    }
}


def classify_word(word):
    lemma = lemmatizer.lemmatize(word.lower())
    for category, keywords in TAXONOMY.items():
        if lemma in keywords:
            return category
    return "Other/General"


def run_systematic_analysis(references, candidates):
    print("Running Systematic Categorical Failure Analysis...")

    # storage for stats per category
    # format: {Category: {'TP': 0, 'FP': 0, 'FN': 0}}
    stats = {cat: {'TP': 0, 'FP': 0, 'FN': 0} for cat in TAXONOMY.keys()}
    stats["Other/General"] = {'TP': 0, 'FP': 0, 'FN': 0}

    for ref_list, cand in zip(references, candidates):
        # 1. Extract Lemmas from Ground Truth (Union of all refs)
        gt_lemmas = set()
        for ref in ref_list:
            tokens = nltk.word_tokenize(ref.lower())
            gt_lemmas.update([lemmatizer.lemmatize(t) for t in tokens if t.isalnum()])

        # 2 extract Lemmas from model supermarket response
        cand_tokens = nltk.word_tokenize(cand.lower())
        cand_lemmas = set([lemmatizer.lemmatize(t) for t in cand_tokens if t.isalnum()])

        # 3 analyze intersection per category
        # check every word in taxonomy to see hallucinated or missed

        # check False Positives (Hallucinations) and True Positives
        for token in cand_lemmas:
            category = classify_word(token)
            if token in gt_lemmas:
                stats[category]['TP'] += 1
            else:
                stats[category]['FP'] += 1  # HALLUCINATION

        # check False Negatives (Misses)
        for token in gt_lemmas:
            category = classify_word(token)
            if token not in cand_lemmas:
                stats[category]['FN'] += 1  # MISS!

    return stats


def print_failure_report(stats):
    print("\n" + "=" * 60)
    print(f"{'FAILURE CATEGORY':<35} | {'HALLUCINATION RATE':<20} | {'OMISSION RATE'}")
    print("=" * 60)

    data_for_csv = []

    for cat, scores in stats.items():
        tp = scores['TP']
        fp = scores['FP']
        fn = scores['FN']

        # Hallucination Rate (FP / (TP + FP)) -> how often is it lying when it speaks about this?
        total_mentions = tp + fp
        hallucination_rate = (fp / total_mentions * 100) if total_mentions > 0 else 0.0

        # Omission Rate (FN / (TP + FN)) -> how often does it miss real things?
        total_real = tp + fn
        omission_rate = (fn / total_real * 100) if total_real > 0 else 0.0

        print(f"{cat:<35} | {hallucination_rate:6.2f}%              | {omission_rate:6.2f}%")

        data_for_csv.append({
            "Category": cat,
            "Hallucination_Rate": hallucination_rate,
            "Omission_Rate": omission_rate,
            "Total_Mentions": total_mentions
        })

    # saving in csv
    df = pd.DataFrame(data_for_csv)

    output_file = r"C:\Users\soban\PycharmProjects\LLaVA\analysis\failure_mode_analysis.csv"
    df.to_csv(output_file, index=False)
    print("\n[Saved details to 'analysis/failure_mode_analysis.csv']")

if __name__ == "__main__":
    GT_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\ground_truth_markets.jsonl"
    MODEL_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\model_responses_ZS_markets.jsonl"

    ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

    statistics = run_systematic_analysis(refs, cands)
    # report in table
    print_failure_report(statistics)
