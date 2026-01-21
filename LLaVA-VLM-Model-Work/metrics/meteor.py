import nltk
import numpy as np

# nltk data check block
# avoid LookupErrors:: check for all necessary nltk resources:
required_packages = ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab']

for package in required_packages:
    try:
        if package == 'wordnet':
            nltk.data.find('corpora/wordnet.zip')
        elif package == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        else:
            nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        print(f"Downloading missing NLTK package: {package}...")
        nltk.download(package)


def compute_meteor(references, candidates):
    scores = []
    #  meteor expect the reference to be tokenized
    #  tokenizing manually

    for ref_list, cand in zip(references, candidates):
        # tokenize references
        ref_tokens = [nltk.word_tokenize(r) for r in ref_list]
        # tokenize candidate
        cand_tokens = nltk.word_tokenize(cand)

        # compute Score
        score = nltk.translate.meteor_score.meteor_score(ref_tokens, cand_tokens)
        scores.append(score)

    return {'METEOR': np.mean(scores)}
