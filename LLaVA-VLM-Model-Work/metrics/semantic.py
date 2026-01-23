import numpy as np
import torch
from bert_score import score
from sentence_transformers import SentenceTransformer, util

#  device : GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Semantic Models on: {device}")

# load SBERT model
# 'all-MiniLM-L6-v2' is fast and effective for similarity
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


def compute_bertscore(references, candidates):

    # computing BERTScore (Precision Recall amd F1) but focus on F1 for paper
    print("Computing BERTScore..")

    # flatten references: BERTScore expects 1:1 or 1:Many.
    # one reference per image 1:1 comparison
    flat_refs = [r[0] for r in references]

    # lang='en' download the default roberta-large model
    P, R, F1 = score(candidates, flat_refs, lang="en", verbose=True, device=device)

    return {
        'BERTScore-F1': F1.mean().item(),
        'BERTScore-Precision': P.mean().item(),
        'BERTScore-Recall': R.mean().item()
    }


def compute_sbert_similarity(references, candidates):
    """
    Computes Cosine Similarity using Sentence-BERT embeddings.
    """
    print("Computing SBERT Semantic Similarity..")

    scores = []

    # encode all candidates in one batch
    cand_embeddings = sbert_model.encode(candidates, convert_to_tensor=True)

    #  loop references because they are list
    for i, ref_list in enumerate(references):
        # encode references for this specific image
        ref_embeddings = sbert_model.encode(ref_list, convert_to_tensor=True)

        # calculate cosine similarity between model generated and refs for this image
        cos_scores = util.cos_sim(cand_embeddings[i], ref_embeddings)
        best_score = torch.max(cos_scores).item()
        scores.append(best_score)

    return {'Semantic-Similarity (SBERT)': np.mean(scores)}
