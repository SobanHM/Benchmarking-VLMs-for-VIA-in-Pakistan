import torch
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os

# verifying gpu for use
device = "cuda" if torch.cuda.is_available() else "cpu"

# load clip model (ViT-B-32 which is standard research baseline)
print(f"Loading CLIP Model on {device}...")
clip_model = SentenceTransformer('clip-ViT-B-32', device=device)


def compute_clipscore(image_dir, image_ids, candidates):
    """
    Computes CLIPScore (Cosine Similarity between Image and Text).
    Args:
        image_dir (str): Path to the folder containing .jpg files
        image_ids (list): List of filenames ['sm_1.jpg', ..]
        candidates (list): List of model generated text strings
    """
    print(f"Computing CLIPScore for {len(image_ids)} images..")

    scores = []
    valid_indices = []  # keep track of which images actually worked

    # 1 encode text (batch processing)
    print("Encoding text captions..")
    text_emb = clip_model.encode(candidates, convert_to_tensor=True)

    # 2 encode images (in looping to handle missing files)
    print("Encoding images..")
    image_embeddings_list = []

    for idx, img_file in enumerate(image_ids):
        img_path = os.path.join(image_dir, img_file)

        try:
            # load and convert image
            img = Image.open(img_path)
            # Encode
            img_emb = clip_model.encode(img, convert_to_tensor=True)
            image_embeddings_list.append(img_emb)
            valid_indices.append(idx)

        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}. Assigning 0 score.")
            # handle missing image by creating a zero-tensor  adn skipping
            #  skip comparison for this index later or assign 0
            image_embeddings_list.append(torch.zeros_like(text_emb[0]))
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            image_embeddings_list.append(torch.zeros_like(text_emb[0]))

    # stacking image embeddings into tensor
    if image_embeddings_list:
        img_emb_tensor = torch.stack(image_embeddings_list)
    else:
        return {'CLIPScore': 0.0}, []

    # 3 compute Cosine Similarity
    # compute diagonal elements (Image_i vs Text_i)
    # util.cos_sim returns a matrix but want diagonal pairs
    cos_scores = util.cos_sim(img_emb_tensor, text_emb)

    # extract score for each valid pair (i, i)
    # match Image to text
    final_scores = []
    for i in range(len(image_ids)):
        score = cos_scores[i][i].item()
        # CLIP raw scores are cosine similarity (-1 to 1).
        # point: sometimes researchers rescale this to 0-100 or 0-2.5 and also 0-1
        # clamp negative scores to 0 for reporting.
        final_scores.append(max(score, 0.0))

    return {'CLIPScore': sum(final_scores) / len(final_scores)}, final_scores
