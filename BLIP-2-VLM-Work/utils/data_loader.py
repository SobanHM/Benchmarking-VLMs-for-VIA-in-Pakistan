import json


def load_data(ground_truth_path, model_response_path):
    """
    Loads and aligns data based on the user's specific JSONL format.
    Expected Keys: 'image' (id) and 'response' (text)
    """

    # function to find text content safely
    def get_text(item):
        # tries common keys in order: 'response', 'text', 'caption', 'description'
        for key in ['response', 'text', 'caption', 'description']:
            if key in item:
                return item[key]
        return ""

    # function to find the ID safely
    def get_id(item):
        #  'image' first then 'image_id'
        for key in ['image', 'image_id', 'id']:
            if key in item:
                return item[key]
        return None

    # load ground truth
    gts = {}
    print(f"Loading Ground Truth from: {ground_truth_path}")
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
                img_id = get_id(item)
                text = get_text(item)

                if img_id is None:
                    print(f"Warning: Line {line_num} in GT has no 'image' key. Skipping.")
                    continue

                if img_id not in gts:
                    gts[img_id] = []
                # handle both single strings and lists of strings
                if isinstance(text, list):
                    gts[img_id].extend(text)
                else:
                    gts[img_id].append(text)
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num} in GT is not valid JSON.")

    # load model responses
    res = {}
    print(f"Loading Model Responses from: {model_response_path}")
    with open(model_response_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
                img_id = get_id(item)
                text = get_text(item)

                if img_id and text:
                    res[img_id] = text
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num} in Model Response is not valid JSON.")

    # align them (intersection of ids)
    references = []
    candidates = []
    ids = []

    for img_id, pred_caption in res.items():
        if img_id in gts:
            ids.append(img_id)
            candidates.append(pred_caption)
            references.append(gts[img_id])  # list of ground truths
        else:
            pass

    print(f"Data Loaded: Found {len(ids)} overlapping images between Ground Truth and Model Responses.")
    return ids, references, candidates
