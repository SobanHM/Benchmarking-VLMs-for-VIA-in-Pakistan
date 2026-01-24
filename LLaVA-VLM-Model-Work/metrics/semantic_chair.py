import sys
import os

# Fix path to find utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from utils.data_loader import load_data

# load SBERT for Semantic Matching
print("Loading SBERT for Semantic Matching...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#  safe list
SAFE_META_WORDS = {
    'image', 'picture', 'photo', 'photograph', 'scene', 'view', 'visual',
    'front', 'back', 'background', 'foreground', 'left', 'right', 'center', 'middle', 'side',
    'top', 'bottom', 'corner', 'edge', 'part', 'area', 'space', 'place', 'spot',
    'shown', 'shows', 'showing', 'seen', 'visible', 'located', 'positioned', 'placed', 'standing', 'sitting',
    'color', 'shape', 'size', 'style', 'type', 'kind', 'look', 'appearance', 'arrangement', 'composition',
    'focus', 'balance', 'depth', 'perspective', 'angle', 'lighting', 'shadow', 'contrast', 'brightness',
    'object', 'item', 'thing', 'element', 'detail', 'feature'
}

def extract_objects(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tagged = nltk.pos_tag(tokens)
    return {lemmatizer.lemmatize(w) for w, t in tagged if t.startswith('NN') and w not in SAFE_META_WORDS}

def compute_semantic_chair(references, candidates, threshold=0.50):
    print(f"Computing SEMANTIC CHAIR (Threshold: {threshold})...")

    hallucinated_objects_count = 0
    total_model_objects_count = 0
    hallucinated_sentences_count = 0

    for ref_list, cand in zip(references, candidates):
        # 1 Get GT Objects
        gt_objects = set()
        for ref in ref_list:
            gt_objects.update(extract_objects(ref))
        gt_list = list(gt_objects)

        # 2 Get Model Objects
        model_objects = extract_objects(cand)
        model_list = list(model_objects)

        # If no objects, skip
        if not model_list: continue

        # 3 Semantic matching
        # Encode all objects
        if not gt_list:
            # if GT has no objects then everything is hallucination
            real_hallucinations = model_list
        else:
            gt_embs = sbert_model.encode(gt_list, convert_to_tensor=True)
            model_embs = sbert_model.encode(model_list, convert_to_tensor=True)

            # calculate Similarity Matrix
            cosine_scores = util.cos_sim(model_embs, gt_embs)

            # check each model object
            real_hallucinations = []
            for i, obj in enumerate(model_list):
                # find best match in GT
                best_match_score = cosine_scores[i].max().item()

                # If match is weak (< threshold), it is hallucination
                # If match is strong (> threshold), it is Synonym (Safe)
                if best_match_score < threshold:
                    real_hallucinations.append(obj)

        # update counts
        num_hallucinated = len(real_hallucinations)
        hallucinated_objects_count += num_hallucinated
        total_model_objects_count += len(model_list)

        if num_hallucinated > 0:
            hallucinated_sentences_count += 1

    # final Calculation
    chair_i = hallucinated_objects_count / total_model_objects_count if total_model_objects_count > 0 else 0
    chair_s = hallucinated_sentences_count / len(candidates) if len(candidates) > 0 else 0

    return chair_i, chair_s
