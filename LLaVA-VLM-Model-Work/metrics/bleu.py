from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def compute_bleu(references, candidates):
    """
    Computes BLEU 1-4.
    references: list of lists of strings
    candidates: list of strings
    """
    scores = {'BLEU-1': [], 'BLEU-2': [], 'BLEU-3': [], 'BLEU-4': []}
    chencherry = SmoothingFunction() # Prevents zero scores for short sentences

    for ref, cand in zip(references, candidates):
        # NLTK expects tokenized inputs (list of words)
        ref_tokens = [r.lower().split() for r in ref]
        cand_tokens = cand.lower().split()

        scores['BLEU-1'].append(sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1))
        scores['BLEU-2'].append(sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1))
        scores['BLEU-3'].append(sentence_bleu(ref_tokens, cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1))
        scores['BLEU-4'].append(sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1))

    # return average bleu scores
    return {k: np.mean(v) for k, v in scores.items()}
