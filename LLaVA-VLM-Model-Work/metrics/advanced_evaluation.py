import sys
import os
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# fixed file path root dirc
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.data_loader import load_data

# ensure resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('stopwords')
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# spatial adn navigation vocab configuration
SPATIAL_TERMS = {
    'left', 'right', 'top', 'bottom', 'center', 'middle', 'front', 'back',
    'behind', 'beside', 'next', 'near', 'above', 'below', 'corner'
}

NAVIGATION_KEYWORDS = {
    'stairs', 'staircase', 'steps', 'ladder',  # Level changes (High Risk)
    'door', 'entrance', 'exit', 'gate',  # Exits (High Risk)
    'path', 'walkway', 'corridor', 'hallway',  # Paths
    'obstacle', 'barrier', 'block',  # Hazards
    'floor', 'ground'  # Surface
}

SAFE_META_WORDS = {
    'image', 'picture', 'photo', 'scene', 'view', 'shown', 'located',
    'positioned', 'visible', 'color', 'shape', 'style', 'appearance'
}


def extract_tokens(text, filter_type=None):
    tokens = nltk.word_tokenize(text.lower())
    if filter_type == 'spatial':
        return {t for t in tokens if t in SPATIAL_TERMS}
    elif filter_type == 'navigation':
        # Simple string matching for lemmas
        lemmas = [lemmatizer.lemmatize(t) for t in tokens]
        return {t for t in lemmas if t in NAVIGATION_KEYWORDS}
    else:
        # standard Object extraction
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        tagged = nltk.pos_tag(tokens)
        return {
            lemmatizer.lemmatize(w)
            for w, t in tagged
            if t.startswith('NN') and w not in SAFE_META_WORDS
        }

def compute_pope_and_specs(references, candidates):
    print("Computing Advanced Metrics (POPE, SPECS, Nav-Safety)...")

    # POPE Metrics (Object Existence)
    tp, fp, fn = 0, 0, 0

    # SPECS Metrics (Spatial Precision)
    spatial_precisions = []
    spatial_recalls = []

    # Navigation Safety
    nav_hallucinations = 0
    nav_mentions_total = 0

    for ref_list, cand in zip(references, candidates):
        # 1 Ground Truth Universe
        gt_objects = set()
        gt_spatial = set()
        gt_nav = set()

        for ref in ref_list:
            gt_objects.update(extract_tokens(ref))
            gt_spatial.update(extract_tokens(ref, 'spatial'))
            gt_nav.update(extract_tokens(ref, 'navigation'))

        # 2 Model Universe
        model_objects = extract_tokens(cand)
        model_spatial = extract_tokens(cand, 'spatial')
        model_nav = extract_tokens(cand, 'navigation')

        # POPE calculation
        # True Positives: In both
        intersection = model_objects.intersection(gt_objects)
        local_tp = len(intersection)
        # False Positives: In model, not in GT (Hallucination)
        local_fp = len(model_objects - gt_objects)
        # False Negatives: In GT, not in model (Omission)
        local_fn = len(gt_objects - model_objects)

        tp += local_tp
        fp += local_fp
        fn += local_fn

        # SPECS (Spatial) calculation
        if model_spatial:
            sp_hits = len(model_spatial.intersection(gt_spatial))
            spatial_precisions.append(sp_hits / len(model_spatial))
        else:
            spatial_precisions.append(1.0)  # no spatial terms = No error

        if gt_spatial:
            sp_hits = len(model_spatial.intersection(gt_spatial))
            spatial_recalls.append(sp_hits / len(gt_spatial))

        # navigational safety
        # did the model mention a 'staircase' that isn't there?
        nav_hallucinations_list = model_nav - gt_nav
        if nav_hallucinations_list:
            nav_hallucinations += len(nav_hallucinations_list)
        nav_mentions_total += len(model_nav)

    # final POPE Stats
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # final Spatial Stats
    avg_spatial_prec = np.mean(spatial_precisions)
    avg_spatial_rec = np.mean(spatial_recalls) if spatial_recalls else 0

    # final Nav Stats
    nav_error_rate = nav_hallucinations / nav_mentions_total if nav_mentions_total > 0 else 0

    return {
        'POPE_Precision': precision,
        'POPE_Recall': recall,
        'POPE_F1': f1,
        'SPECS_Spatial_Precision': avg_spatial_prec,
        'SPECS_Spatial_Recall': avg_spatial_rec,
        'Nav_Hallucination_Rate': nav_error_rate
    }


if __name__ == "__main__":
    GT_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\ground_truth_markets.jsonl"
    MODEL_PATH = r"C:\Users\soban\PycharmProjects\LLaVA\data\model_responses_ZS_markets.jsonl"

    ids, refs, cands = load_data(GT_PATH, MODEL_PATH)

    results = compute_pope_and_specs(refs, cands)

    print("\nIntelligent Evaluation Results:====")
    print(f"[POPE] Object Existence F1:   {results['POPE_F1']:.4f}")
    print(f"[POPE] Precision (Truth):     {results['POPE_Precision']:.4f}")
    print(f"[POPE] Recall (Coverage):     {results['POPE_Recall']:.4f}")
    print("-" * 40)
    print(f"[SPECS] Spatial Precision:    {results['SPECS_Spatial_Precision']:.4f}")
    print(f"        (How often 'left/right' is correct)")
    print("-" * 40)
    print(f"[SAFETY] Navigation Hallucination Rate: {results['Nav_Hallucination_Rate'] * 100:.1f}%")
    print(f"         (Percentage of navigation terms that were fake)")
    print("================================================")
