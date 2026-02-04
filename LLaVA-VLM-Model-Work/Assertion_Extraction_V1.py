"""
Assertion Extraction v1.1
Author: Soban Hussain
Purpose:
Refined extraction of atomic assertions from zero-shot VLM outputs
for scene narration and safe navigation (VIA).

Fixes applied:
1. Filter existence assertions
2. Anchor spatial assertions to entities
3. Merge safety + existence
4. Action subtypes
"""

import json
import re
from tqdm import tqdm
import spacy

from Assertions_Ontology import (
    ENTITY_ONTOLOGY,
    PRODUCT_CATEGORIES,
    SAFETY_DIMENSIONS,
    SAFETY_RULES,
    SPATIAL_TERMS,
    NAVIGATION_VERBS,
)

nlp = spacy.load("en_core_web_sm")

# helper utilities
def normalize_entity(token_text):
    t = token_text.lower()
    if t in ENTITY_ONTOLOGY:
        return t
    for cat, items in PRODUCT_CATEGORIES.items():
        if t in items:
            return t
    return None


def is_safety_entity(entity):
    for dim, items in SAFETY_DIMENSIONS.items():
        if entity in items:
            return dim
    return None


def action_subtype(sentence):
    s = sentence.lower()
    if any(v in s for v in ["avoid", "careful"]):
        return "A_avoid"
    if any(v in s for v in ["stop", "wait"]):
        return "A_stop"
    if any(v in s for v in ["turn"]):
        return "A_turn"
    if any(v in s for v in ["follow", "go along"]):
        return "A_follow"
    return "A_move"


# Core extraction
def extract_assertions(text, image_id):
    doc = nlp(text)
    assertions = []
    idx = 0

    for sent in doc.sents:
        sent_text = sent.text.lower()

        # --- ENTITY COLLECTION ---
        entities = []
        for token in sent:
            if token.pos_ == "NOUN":
                ent = normalize_entity(token.text)
                if ent:
                    entities.append(ent)

        # Existence + Safety-Existence
        for ent in set(entities):
            safety_dim = is_safety_entity(ent)

            if safety_dim:
                assertions.append({
                    "assertion_id": f"{image_id}_SE_{idx}",
                    "type": "SE",
                    "entity": ent,
                    "safety_dimension": safety_dim,
                    "criticality": "C1"
                })
            else:
                assertions.append({
                    "assertion_id": f"{image_id}_E_{idx}",
                    "type": "E",
                    "entity": ent,
                    "criticality": "C3"
                })
            idx += 1

        # Spatial (entity-anchored)
        for rel in SPATIAL_TERMS:
            if rel in sent_text:
                for ent in entities:
                    assertions.append({
                        "assertion_id": f"{image_id}_S_{idx}",
                        "type": "S",
                        "entity": ent,
                        "relation": rel,
                        "statement": sent.text,
                        "criticality": "C2"
                    })
                    idx += 1

        # Action (subtyped)
        if any(v in sent_text for v in NAVIGATION_VERBS):
            assertions.append({
                "assertion_id": f"{image_id}_A_{idx}",
                "type": "A",
                "subtype": action_subtype(sent_text),
                "instruction": sent.text,
                "criticality": "C1"
            })
            idx += 1

    return assertions

# IO Pipeline
def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        return json.load(f)


def run_pipeline(input_path, output_path):
    data = load_input(input_path)
    results = []

    for item in tqdm(data, desc="Extracting assertions"):
        image_id = item["image"]
        response = item["response"]
        assertions = extract_assertions(response, image_id)
        results.append({
            "image_id": image_id,
            "assertions": assertions
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Assertions saved to: {output_path}")


if __name__ == "__main__":
    input_file = r"C:\Users\soban\Downloads\via_markets_dataset_zs_llava.jsonl"
    output_assertions = "assertions_supermarket_v1_1.json"
    print("Pipeline Ready for assertion extraction.")
    run_pipeline(input_file, output_assertions)

