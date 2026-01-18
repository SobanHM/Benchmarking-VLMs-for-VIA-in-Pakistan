# implememted on dual 8 GB VRAM RTX Super 2080 GPU
import os
import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

IMAGE_FOLDER = r"C:\Users\soban\Downloads\VLM_Evaluation_Benchmark_Dataset_FYP\Pakistani_Supermarkets_Dataset"
SAVE_FILE = r"C:\Users\soban\Downloads\response_llava_supermarket_via.jsonl"

SYSTEM_PROMPT = """
You are a visual assistant for a blind or visually impaired person.
Describe exactly what is visible in the image using clear, simple, natural human language.

Internal rules (DO NOT show them in the response):
- Describe what the person is facing from their point of view.
- Mention important objects and their positions (left, right, front, front-right, front-left, behind).
- Mention distances only if clearly visible.
- Mention useful landmarks for orientation.
- Warn about obstacles or hazards.
- Give short guidance if the user may want to reach something.
- Keep it short, natural, and avoid guessing anything not visible.
- Speak as if you are physically standing there guiding the user.

Do NOT use labels, bullet points, or structured formatting.
Just speak naturally.
"""


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
QUERY = "I am visually impaired person. I cannot see anything. Describe this image correctly in a helpful human way. Guideline, Describe the environment from my point of view"
# QUERY = "I am a blind person and cannot see anything in front of me. You are my visual assistant. Please describe the scene exactly as it appears in the image, based only on what you truly see. Avoid guessing or assuming anything that is not visible. Describe the environment from my point of view, as if I am standing there. Explain what objects, people, places, activities or anything are in front of me, including their positions and distances if possible. Use clear, simple, natural language that sounds human and helpful"

print("Loading model...")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = LlavaForConditionalGeneration.from_pretrained( MODEL_ID, device_map="auto", dtype=dtype )
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

results = []
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

print(f"Found {len(image_files)} images.")

for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)

    print(f"Processing: {img_name}")

    image = Image.open(img_path).convert("RGB")

    # prepare messages with system prompt + image + user query
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": QUERY},
            ],
        }
    ]

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": QUERY},
    #         ],
    #     }
    # ]

    # create chat text
    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # align text + image
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # ensure config ok
    gc = model.generation_config
    if gc.pad_token_id is None:
        gc.pad_token_id = processor.tokenizer.pad_token_id
    if gc.eos_token_id is None:
        gc.eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
            top_p=0.9,
            do_sample=True
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    entry = {
        "image": img_name,
        "query": QUERY,
        "response": response
    }
    results.append(entry)

    # append to json file
    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("\nDataset saved to:", SAVE_FILE)
