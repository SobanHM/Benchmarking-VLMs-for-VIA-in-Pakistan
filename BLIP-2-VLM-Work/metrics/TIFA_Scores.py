import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoProcessor, AutoModelForVision2Seq

# nltk.download("punkt")
# ---------------- CONFIG ----------------
JSONL_FILE = "BLIP_2/data/blip_model_responses_ZS_markets.jsonl"
IMAGE_DIR = "via-sn-dataset/pakistani_supermarket/"
OUTPUT_CSV = "tifa_results_blip_2_sm.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VQA model (officially supported style)
VQA_MODEL = "llava-hf/llava-1.5-7b-hf"
# ----------------------------------------

print("[INFO] Loading VQA model...")
processor = AutoProcessor.from_pretrained(VQA_MODEL)
model = AutoModelForVision2Seq.from_pretrained(
    VQA_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def caption_to_questions(caption):
    """
    Split caption into factual statements (sentences)
    Convert each into a yes/no verification question
    """
    sentences = sent_tokenize(caption)
    questions = []

    for s in sentences:
        s = s.strip()
        if len(s) < 5:
            continue
        q = f"Is this true: {s}?"
        questions.append(q)

    return questions

def ask_vqa(image, question):
    """
    Ask VQA model a yes/no question
    """
    prompt = f"<image>\nUSER: {question}\nASSISTANT:"
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.lower()

def answer_is_yes(answer):
    """
    Decide if model supports the claim
    """
    yes_words = ["yes", "true", "correct", "indeed"]
    no_words = ["no", "not", "false", "incorrect"]

    for y in yes_words:
        if y in answer:
            return True
    for n in no_words:
        if n in answer:
            return False

    return False  # conservative

results = []

print("[INFO] Starting TIFA evaluation..")

with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        data = json.loads(line)

        image_file = data["image"]
        caption = data["response"]

        image_path = os.path.join(IMAGE_DIR, image_file)
        if not os.path.exists(image_path):
            continue

        image = Image.open(image_path).convert("RGB")

        questions = caption_to_questions(caption)

        if len(questions) == 0:
            continue

        supported = 0

        for q in questions:
            answer = ask_vqa(image, q)
            if answer_is_yes(answer):
                supported += 1

        tifa_score = supported / len(questions)

        results.append({
            "image": image_file,
            "tifa_score": tifa_score,
            "num_facts": len(questions)
        })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print("\n[INFO] Evaluation completed!")
print(f"[INFO] Results saved to: {OUTPUT_CSV}")
print(f"[RESULT: BLIP_2 sm] Mean TIFA Score: {df['tifa_score'].mean():.4f}")
