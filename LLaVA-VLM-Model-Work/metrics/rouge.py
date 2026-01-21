from rouge_score import rouge_scorer
import numpy as np

def compute_rouge(references, candidates):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for ref_list, cand in zip(references, candidates):
        # rouge usually compares 1-to-1 so i  take the max score across all references for a single image
        instance_scores = [scorer.score(ref, cand)['rougeL'].fmeasure for ref in ref_list]
        scores.append(max(instance_scores))

    return {'ROUGE-L': np.mean(scores)}
