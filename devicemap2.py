from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import infer_auto_device_map
import torch
import json
from huggingface_hub import login  # Importing the login function

# Step 1: Authenticate with Hugging Face
login(token="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")  # Replace with your actual Hugging Face token

MODEL_ID = "meta-llama/Meta-Llama-3-8B"  # ✅ Correct model identifier

# Step 2: Configure quantization options
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # Enables CPU offloading of certain layers
)

# Step 3: Load model temporarily on CPU to analyze the device map structure
print("Loading model to infer device map...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",  # Start on CPU, will be adjusted later
    quantization_config=bnb_config,
    trust_remote_code=True,  # Trust remote code for model loading
    torch_dtype=torch.float16  # Load in half precision to save memory
)

# Step 4: Define memory limits and policy for splitting the model across devices
device_map = infer_auto_device_map(
    model,
    max_memory={
        0: "11GiB",       # GPU: 11GB out of 12GB
        "cpu": "12GiB",    # CPU: 12GB out of 16GB
    },
    no_split_module_classes=["LlamaBlock"]  # Ensures LlamaBlock is treated as a whole
)

# Step 5: Save the device map to a JSON file
with open("device_map_llama3_8b.json", "w") as f:
    json.dump(device_map, f, indent=2)

print("✅ Device map generated and saved as device_map_llama3_8b.json")



