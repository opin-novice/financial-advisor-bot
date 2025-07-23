from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch
import json

MODEL_ID = "Qwen/Qwen3-8B"  # ✅ Or Qwen3-14B if you're brave

# Step 1: Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Step 2: Temporarily load model on CPU just to analyze structure
print("Loading model to infer device map...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",  # We’ll split it ourselves
    quantization_config=bnb_config,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# Step 3: Define memory limits and split policy
device_map = infer_auto_device_map(
    model,
    max_memory={
        0: "11GiB",       # GPU: 11GB out of 12GB
        "cpu": "12GiB",   # CPU: 12GB out of 16GB
    },
    no_split_module_classes=["QWenBlock"]  # ✅ Ensures proper splitting
)

# Step 4: Save the device map
with open("device_map.json", "w") as f:
    json.dump(device_map, f, indent=2)

print("✅ Device map generated and saved as device_map.json")
