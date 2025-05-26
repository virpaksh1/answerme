import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import sys
import os

try:
    import bitsandbytes as bnb
    print("Successfully imported bitsandbytes")
except ImportError:
    print("Error importing bitsandbytes. Attempting to install again...")
    import bitsandbytes as bnb


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Qwen1.5-7B-Chat model - publicly available and efficient to run in Google Colab with T4 GPU
model_name = "Qwen/Qwen1.5-7B-Chat"

print(f"Loading {model_name}...")
start_time = time.time()

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Trying to load the model with 4-bit quantization for efficiency
try:
    print("Attempting to load model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        device_map="auto",
        trust_remote_code=True,
        quantization_config={"load_in_4bit": True}  # 4-bit quantization for memory efficiency
    )
except Exception as e:
    print(f"4-bit quantization failed with error: {str(e)}")
    print("Falling back to 8-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True  # Try 8-bit quantization instead
        )
    except Exception as e2:
        print(f"8-bit quantization failed with error: {str(e2)}")
        print("Falling back to standard loading (will use more memory)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

load_time = time.time() - start_time
print(f"Model loaded within {load_time:.2f} seconds")

def generate_response(user_input, chat_history=None):
    if chat_history is None:
        chat_history = []

    system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should be engaging and fun.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information, you are also a motor driver ,
if a table with information is provided answer accordingly."""

    # Formatting the conversation for the model
    messages = [{"role": "system", "content": system_prompt}]

    # Adding chat history for a full context of the conversation
    for message in chat_history:
        messages.append(message)

    # Adding the current user input
    messages.append({"role": "user", "content": user_input})

    # Tokenization: converting messages to model input format
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generating response: this part may take some time to execute at first
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # Decoding the generated response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extracting only the assistant's response, instead of the full raw output
    assistant_response = full_response.split(user_input)[-1].strip()

    # Further cleaning up the response if it contains role markers or other artifacts
    if "assistant" in assistant_response.lower()[:20]:
        assistant_response = assistant_response.split(":", 1)[-1].strip()

    return assistant_response