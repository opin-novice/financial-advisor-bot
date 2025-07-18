from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json
MODEL_ID = "Qwen/Qwen3-14B"
SAVE_PATH = "./quantized_model"

# Load optimized device_map
with open("device_map.json") as f:
    device_map = json.load(f)

# Only quantize and save if not already saved
if not os.path.exists(SAVE_PATH):
    print("Quantizing and saving Qwen3-14B...")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype="auto",
        trust_remote_code=True
    )

    # Save both model and tokenizer
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"Model quantized and saved to {SAVE_PATH}")
else:
    print(f"Quantized model already exists at {SAVE_PATH}")