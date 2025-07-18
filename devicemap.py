from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch

MODEL_ID = "Qwen/Qwen3-14B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load model temporarily with no weights to analyze structure
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",  # temporarily on CPU to avoid loading to GPU
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

device_map = infer_auto_device_map(
    model,
    max_memory={
        0: "12GiB",       # Your GPU
        "cpu": "16GiB",   # Your system RAM
    },
    no_split_module_classes=["QWenBlock"]  # Qwen3 uses QWenBlock transformer layers
)

# Print or save the map
import json
with open("device_map.json", "w") as f:
    json.dump(device_map, f, indent=2)

print("✅ Optimized device map generated and saved to device_map.json")