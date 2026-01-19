import os
import json
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

IMAGE_FOLDER = (r"C:\Users\soban\Downloads\VLM_Evaluation_Benchmark_Dataset_FYP\Pakistani_Supermarkets_Dataset")
SAVE_FILE = r"C:\Users\soban\Downloads\LLaVA_Fewshot_VIA_Supermarket-split.jsonl"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=dtype
)

processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)

# few-shot template of VIA (designed after literature review)
FEWSHOT_PROMPT = """
You are a visual assistant for blind person. Always follow these rules:

1. Describe ONLY what is truly visible. No assumptions.
2. Always describe from the user's point of view.
3. Use short, clear, human sentences.
4. Never repeat sentences.
5. Never mention objects unless you can clearly see them.
6. Focus on:
   - Orientation (front side, left side, right)
   - What is in Left side?
   - What is in right side?
   - What is in front view?
   - Items in image exactly.
   - Distances (near, far, steps)
   - Path safety and obstacles
   - Only key objects
7. Maximum 7 sentences.

### EXAMPLE 1
Assistant:
You’re inside a small shop, facing a glass door that leads outside. The space in front of you
is mostly open. On your left, very close, a man is standing at a counter. On your right, slightly farther away on the floor, there are suitcases.
If you walk forward three to four small steps, you’ll reach the doorway to exit. There’s a slight raised threshold at the bottom—approach slowly and
check it with your cane. Once outside, you’ll step onto a small platform facing parked cars.

### Example 2
You’re facing the shop’s front door. The space ahead is open for three small steps. There is small staris up to the door. Be careful.
A man is close on your left beside a counter. A metal basket is on the floor to your right.
Move straight to reach the doorway; watch for a small raised edge at the bottom.

### EXAMPLE 3
You’re in a supermarket aisle with shelves on both sides. There are apples, oranges, and vegetables in baskets.
A person stands slightly ahead on your right. Shift a bit left to move forward.
The floor is open and smooth. After a few steps, the aisle widens and there are baskets for fruits and vegetables.

### EXAMPLE 4
Assistant:
You’re standing in a supermarket aisle with shelves on both your left and right filled with
colorful packets of powdered milk and cereal. Directly ahead of you but slightly to your
right, a person is standing holding a white shopping bag. You can continue forward if you
shift a bit left to avoid him. The floor is smooth and clear. After around four to six
steps forward, the aisle opens up again and gives you more space.

### Now DESCRIBE the USER'S IMAGE in the same style.
"""

results = []
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"Found {len(image_files)} images.")

for img_name in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    print(f"\nProcessing: {img_name}")

    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": FEWSHOT_PROMPT}
            ]
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    # fix pad token issues (if not fixed then model response is empty)
    gen_cfg = model.generation_config
    gen_cfg.pad_token_id = processor.tokenizer.pad_token_id
    gen_cfg.eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False, # prevent hallucination           
            top_p = None,
            temperature = None,
            repetition_penalty=1.3,    # added prevent loops
        )

    # extract only the generated tokens
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True
    )[0].strip()

    entry = {
        "image": img_name,
        "query": "VIA Few-Shot Description",
        "response": response
    }

    results.append(entry)

    with open(SAVE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("\nDataset saved to:", SAVE_FILE)
