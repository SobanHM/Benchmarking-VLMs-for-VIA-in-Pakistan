from pycocoevalcap.cider.cider import Cider

def compute_cider(ids, references, candidates):

    # cider requires a dictionary format: {id: [captions]}
    # formatting for pycocoevalcap
    gts = {str(i): [{'response': r} for r in refs] for i, refs in zip(ids, references)}
    res = [{'image': str(i), 'response': c} for i, c in zip(ids, candidates)]

    # run Scorer
    scorer = Cider()
    # cider scorer expects a simpler dict for gts if using the generic interface [class:??]
    # format exactly as library expects for direct calls:
    gts_simple = {str(i): refs for i, refs in zip(ids, references)}
    res_simple = {str(i): [c] for i, c in zip(ids, candidates)}

    score, scores = scorer.compute_score(gts_simple, res_simple)

    # score is the average bcs scores is the list per image
    return {'CIDEr': score}, scores
  
