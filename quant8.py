from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import json

MODEL_ID = "Qwen/Qwen3-8B"  # ✅ Use Qwen3-8B model
SAVE_PATH = "./quantized_qwen3_8b"

# Load device map from file
with open("device_map.json") as f:
    device_map = json.load(f)

# Only quantize and save if not already saved
if not os.path.exists(SAVE_PATH):
    print("Quantizing and saving Qwen3-8B...")

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # You can adjust this to 3.0 for more aggressive quantization
        llm_int8_enable_fp32_cpu_offload=True
    )

    # Load tokenizer and quantized model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,  # ✅ Use your custom map here
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    # Save quantized model and tokenizer
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"✅ Qwen3-8B quantized and saved to {SAVE_PATH}")
else:
    print(f"✅ Quantized Qwen3-8B already exists at {SAVE_PATH}")
