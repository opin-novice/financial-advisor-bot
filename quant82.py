from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
import torch
from huggingface_hub import login  # Importing the login function

# Step 1: Authenticate with Hugging Face
login(token="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")  # Replace with your actual Hugging Face token

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
SAVE_PATH = "./quantized_llama3_8b"

# Load device map
with open("device_map_llama3_8b.json") as f:
    device_map = json.load(f)

# Check if already saved
if not os.path.exists(SAVE_PATH):
    print("Quantizing and saving Meta-LLaMA-3-8B...")

    # BitsAndBytes quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=False
    )

    # Save
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"✅ Meta-LLaMA-3-8B quantized and saved to {SAVE_PATH}")
else:
    print(f"✅ Quantized model already exists at {SAVE_PATH}")
