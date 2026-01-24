import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# ensure resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def extract_objects(text):
    """
    Extracts nouns (objects) from text using POS tagging.
    Returns a set of lemmatized nouns.
    """
    tokens = nltk.word_tokenize(text.lower())
    # filter out stopwords and non-alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # pos tagging
    tagged = nltk.pos_tag(tokens)

    # keep only nouns (NN, NNS, NNP ..)
    objects = {lemmatizer.lemmatize(word) for word, tag in tagged if tag.startswith('NN')}

    return objects


def compute_chair(references, candidates):
    """
    Computes Approximate CHAIR metrics (Caption Hallucination Assessment with Image Relevance).
    use the Ground Truth Description nouns as the "Safe List".
    """
    print("Computing CHAIR (Hallucination) Metrics...")

    hallucinated_objects_count = 0
    total_model_objects_count = 0
    hallucinated_sentences_count = 0
    total_sentences_count = len(candidates)

    chair_details = []

    for ref_list, cand in zip(references, candidates):
        # 1 build "Ground Truth Universe" from all human references
        gt_objects = set()
        for ref in ref_list:
            gt_objects.update(extract_objects(ref))

        # 2 extract model objects
        model_objects = extract_objects(cand)

        # 3 find hallucinations (Model has it, GT does not)
        # Note: this is Approximate because if GT says drink and model says coke
        # it might flag "coke" as false. this is a valid Consistency Check
        hallucinations = [obj for obj in model_objects if obj not in gt_objects]

        # ypdate counts
        num_hallucinated = len(hallucinations)
        num_total = len(model_objects)

        hallucinated_objects_count += num_hallucinated
        total_model_objects_count += num_total

        if num_hallucinated > 0:
            hallucinated_sentences_count += 1

        chair_details.append({
            'CHAIR_Score': 1 if num_hallucinated > 0 else 0,
            'Hallucinated_Objects': hallucinations
        })

    # computing final scores
    chair_i = hallucinated_objects_count / total_model_objects_count if total_model_objects_count > 0 else 0
    chair_s = hallucinated_sentences_count / total_sentences_count if total_sentences_count > 0 else 0

    return {
        'CHAIR-i (Object Error Rate)': chair_i,
        'CHAIR-s (Sentence Error Rate)': chair_s
    }, chair_details
