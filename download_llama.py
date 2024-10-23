from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import dotenv

# Load environment variables
dotenv.load_dotenv(".env")
token = os.getenv("HF_TOKEN")

# Log in to Hugging Face Hub
login(token=token)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
local_model_dir = "Llama-3.2-3B-Instruct"

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.save_pretrained(local_model_dir)

# Download model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
model.save_pretrained(local_model_dir)
