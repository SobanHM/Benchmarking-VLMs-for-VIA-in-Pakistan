# utils/text_filters.py

def spatial_filter(text, nlp):
    keywords = {"left", "right", "ahead", "forward", "behind", "meters", "steps"}
    return " ".join(
        t.text for t in nlp(text)
        if t.text.lower() in keywords or t.pos_ == "NUM"
    )


def object_filter(text, nlp):
    return " ".join(t.text for t in nlp(text) if t.pos_ in ["NOUN", "PROPN"])


def hazard_filter(text, nlp):
    hazards = {"obstacle", "crowd", "wet", "slippery", "blocked", "person", "cart"}
    return " ".join(
        t.text for t in nlp(text)
        if t.text.lower() in hazards
    )


def action_filter(text, nlp):
    return " ".join(t.text for t in nlp(text) if t.pos_ == "VERB")


def context_filter(text, nlp):
    contexts = {"market", "aisle", "checkout", "shop", "street", "store"}
    return " ".join(
        t.text for t in nlp(text)
        if t.text.lower() in contexts
    )


def generic_filter(text, nlp):
    return " ".join(
        t.text for t in nlp(text)
        if t.pos_ in ["NOUN", "VERB"]
    )

def dimension_generic_filter(text, nlp, dimension):
    """
    Create a dimension-aware generic baseline by preserving
    only the minimal semantic core required for that dimension.
    """
    doc = nlp(text)
    tokens = []

    for token in doc:
        # ---------- SPATIAL ----------
        if dimension == "spatial":
            # Keep verbs only (remove left/right/distance)
            if token.pos_ == "VERB":
                tokens.append(token.text)

        # ---------- OBJECT ----------
        elif dimension == "object":
            # Keep object nouns only
            if token.pos_ in ["NOUN", "PROPN"]:
                tokens.append(token.text)

        # ---------- HAZARD ----------
        elif dimension == "hazard":
            # Keep hazard nouns (obstacle, crowd)
            if token.pos_ == "NOUN":
                tokens.append(token.text)

        # ---------- ACTION ----------
        elif dimension == "action":
            # Keep verbs, remove ordering and step numbers
            if token.pos_ == "VERB":
                tokens.append(token.text)

        # ---------- CONTEXT ----------
        elif dimension == "context":
            # Keep scene-level nouns only
            if token.pos_ in ["NOUN", "PROPN"]:
                tokens.append(token.text)

    return " ".join(tokens)
