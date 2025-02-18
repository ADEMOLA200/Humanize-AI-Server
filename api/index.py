import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import login
import torch

# Force CPU optimization
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment")

app = Flask(__name__)

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        login(HF_TOKEN)
        model = T5ForConditionalGeneration.from_pretrained(
            "t5-small",
            token=HF_TOKEN,
            load_in_8bit=True,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = T5Tokenizer.from_pretrained(
            "t5-small",
            token=HF_TOKEN
        )

@app.route('/')
def health_check():
    return "Server operational! Model status: " + ("Loaded" if model else "Not loaded")

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    load_model()
    
    text = request.json['text']
    inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=768, truncation=True)
    
    outputs = model.generate(
        inputs,
        max_length=768,
        min_length=200,
        do_sample=True,
        num_beams=3,  # Reduced from 5 to save memory
        temperature=0.7,
        top_p=0.9
    )
    
    return jsonify({'paraphrased': tokenizer.decode(outputs[0], skip_special_tokens=True)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
