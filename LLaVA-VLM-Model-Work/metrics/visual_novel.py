import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import numpy as np

# ensure nltk sentence splitter is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = SentenceTransformer('clip-ViT-B-32', device=device)


def compute_sliding_clipscore(image_dir, image_ids, candidates):
    """
    Novelty Metric: Computes CLIPScore using a Sliding Window to avoid 77-token truncation.
    We split text into sentences and take the MAX similarity with the image.
    """
    print(f"Computing Novel Sliding-Window CLIPScore on {device}...")

    scores = []

    for img_id, text in zip(image_ids, candidates):
        img_path = os.path.join(image_dir, img_id)

        try:
            # 1 load image
            img = Image.open(img_path)
            img_emb = clip_model.encode(img, convert_to_tensor=True)

            # 2 Novelty: split Text into sentences
            # this avoids the 77 token limit of standard CLIP
            segments = nltk.sent_tokenize(text)
            if not segments: segments = [text]

            # 3 encode all segments
            seg_embs = clip_model.encode(segments, convert_to_tensor=True)

            # 4 compute similarity (image vs all segments)
            # find the segment that best matches the image
            cos_scores = util.cos_sim(img_emb, seg_embs)
            best_score = torch.max(cos_scores).item()

            scores.append(max(best_score, 0.0))

        except Exception as e:
            print(f"Error processing {img_id}: {e}")
            scores.append(0.0)

    return {'Sliding_CLIPScore': np.mean(scores)}, scores
