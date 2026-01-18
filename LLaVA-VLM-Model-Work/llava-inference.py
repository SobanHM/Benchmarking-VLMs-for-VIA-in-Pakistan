# Inference is run and tested on Dual 8GB VRAM RTX Super GeForce 2080 nvidia GPU
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
# used the hf-converted llava model
# IMAGE_PATH = r"C:\Users\soban\OneDrive\Pictures\sadar_market_rawalpindi_shoesshop.jpg"
IMAGE_PATH = r"C:\Users\soban\Downloads\VLM_Evaluation_Benchmark_Dataset\Pakistani_Supermarkets_Dataset\sm_29.jpg"
#    "Hello, I am blind person and need your assistance to complete the task of Approaching the Shoe shop. The target is to go Shoes Shop. Please give step-by-step directions and distance guidance based on the scene to reach it safely."
PROMPT = (
 "I am visually impaired person. I can not see. Describe this image correctly from my position."
)

# load model & processor
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print("Loading model:", MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained( MODEL_ID, dtype=dtype, device_map="auto")
processor = AutoProcessor.from_pretrained( MODEL_ID,
    use_fast=True,  # avoid sentencepiece and protobuf
)

image = Image.open(IMAGE_PATH).convert("RGB")

# chat with model
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }
]

#  tokenize=False -> returns string
chat_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

# prepare text and image together and it returns aligned input_ids + pixel_values.
# wrap in lists so batch dimension = 1 is explicit
inputs = processor(
    text=[chat_text],
    images=[image],
    return_tensors="pt",
).to(model.device)

# ensure generation config has pad/eos tokens
gc = model.generation_config
if getattr(gc, "pad_token_id", None) is None:
    gc.pad_token_id = processor.tokenizer.pad_token_id
if getattr(gc, "eos_token_id", None) is None:
    gc.eos_token_id = processor.tokenizer.eos_token_id

# generating description
print("generating description: ")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,   # so temperature/top_p is enterntained
        temperature=0.2,
        top_p=0.9,
    )

# Decodi only the new tokens for cleaner scene naration
new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

print("\nuser query:", PROMPT)
print("description:", answer)

