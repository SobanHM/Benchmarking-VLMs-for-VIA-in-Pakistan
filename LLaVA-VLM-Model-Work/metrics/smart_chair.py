import sys
import os

# fix: add the project root to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from utils.data_loader import load_data

# Ensure resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- THE SMART FILTER ---
# These words are NOT hallucinations; they are just "Style"
SAFE_META_WORDS = {
    'image', 'picture', 'photo', 'photograph', 'scene', 'view', 'visual',
    'front', 'back', 'background', 'foreground', 'left', 'right', 'center', 'middle', 'side',
    'top', 'bottom', 'corner', 'edge', 'part', 'area', 'space', 'place', 'spot',
    'shown', 'shows', 'showing', 'seen', 'visible', 'located', 'positioned', 'placed', 'standing', 'sitting',
    'color', 'shape', 'size', 'style', 'type', 'kind', 'look', 'appearance', 'arrangement', 'composition',
    'focus', 'balance', 'depth', 'perspective', 'angle', 'lighting', 'shadow', 'contrast', 'brightness',
    'object', 'item', 'thing', 'element', 'detail', 'feature'
}


def extract_objects_smart(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # POS Tagging
    tagged = nltk.pos_tag(tokens)

    # Keep Nouns (NN) that are NOT in the Safe List
    objects = {
        lemmatizer.lemmatize(word)
        for word, tag in tagged
        if tag.startswith('NN') and word not in SAFE_META_WORDS
    }

    return objects


def compute_smart_chair(references, candidates):
    print("Computing SMART CHAIR (Filtering Meta-Words)...")

    hallucinated_objects_count = 0
    total_model_objects_count = 0
    hallucinated_sentences_count = 0
    total_sentences_count = len(candidates)

    for ref_list, cand in zip(references, candidates):
        # 1. Ground Truth Universe
        gt_objects = set()
        for ref in ref_list:
            gt_objects.update(extract_objects_smart(ref))

        # 2. Model Objects (Filtered)
        model_objects = extract_objects_smart(cand)

        # 3. Find REAL Hallucinations
        hallucinations = [obj for obj in model_objects if obj not in gt_objects]

        num_hallucinated = len(hallucinations)

        hallucinated_objects_count += num_hallucinated
        total_model_objects_count += len(model_objects)

        if num_hallucinated > 0:
            hallucinated_sentences_count += 1

    # Compute Final Scores
    chair_i = hallucinated_objects_count / total_model_objects_count if total_model_objects_count > 0 else 0
    chair_s = hallucinated_sentences_count / total_sentences_count if total_sentences_count > 0 else 0

    return chair_i, chair_s
