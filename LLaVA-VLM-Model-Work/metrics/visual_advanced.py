import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import numpy as np

# ensure nltk sentence splitter is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = SentenceTransformer('clip-ViT-B-32', device=device)


def compute_sliding_window_clipscore(image_dir, image_ids, candidates, window_type='sentence'):
    """
    Computes 'Dense CLIPScore' by splitting text into segments.
    Returns:
        avg_score (float): Mean of the Max-Segment scores across dataset.
        details (list): List of scores per image.
    """
    print(f"Computing Sliding Window CLIPScore ({window_type}-level) on {device}...")

    final_scores = []

    for img_id, text in zip(image_ids, candidates):
        img_path = os.path.join(image_dir, img_id)

        # 1. Load Image
        try:
            img = Image.open(img_path)
            img_emb = clip_model.encode(img, convert_to_tensor=True)
        except Exception as e:
            print(f"Error loading {img_id}: {e}")
            final_scores.append(0.0)
            continue

        # 2. Split Text into Segments
        if window_type == 'sentence':
            segments = nltk.sent_tokenize(text)
        else:
            # fallback: sliding window of 50 words with 10 word overlap
            words = text.split()
            segments = []
            chunk_size = 50
            overlap = 10
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                segments.append(chunk)

        if not segments:
            segments = [text]

        # 3 encode  segments
        seg_embs = clip_model.encode(segments, convert_to_tensor=True)

        # 4 compute Similarity against each image
        # resulting shape: [num_segments]
        cos_scores = util.cos_sim(img_emb, seg_embs)[0]

        # 5 novel aggregation strategy
        # we take the MAX score
        # Rationale: If any sentence, accurately describes the scene the model saw it.
        # "You are standing in a room" -> score 0.15 (Low)
        # "There is a red chair" -> score 0.45 (High) -> THIS is the signal we want.
        max_score = torch.max(cos_scores).item()

        final_scores.append(max(max_score, 0.0))

    return np.mean(final_scores), final_scores
