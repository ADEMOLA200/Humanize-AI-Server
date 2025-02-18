import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import login
import torch

# Force CPU optimization and disable unnecessary features
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Reduces logging noise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            logger.info("Loading model...")
            login(HF_TOKEN)
            model = T5ForConditionalGeneration.from_pretrained(
                "t5-small",
                token=HF_TOKEN,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            tokenizer = T5Tokenizer.from_pretrained(
                "t5-small",
                token=HF_TOKEN
            )
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

@app.route('/')
def health_check():
    try:
        load_model()
        return "Server operational! Model status: Loaded"
    except Exception as e:
        return f"Server operational but model failed to load: {str(e)}"

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    try:
        load_model()
        
        text = request.json.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400
            
        inputs = tokenizer.encode("paraphrase: " + text, 
                                return_tensors="pt", 
                                max_length=512,  # Reduced from 768
                                truncation=True)
        
        outputs = model.generate(
            inputs,
            max_length=256,  # Reduced output length
            min_length=50,   # More reasonable minimum
            do_sample=True,
            num_beams=2,    # Further reduced
            temperature=0.7,
            top_p=0.9,
            early_stopping=True
        )
        
        return jsonify({'paraphrased': tokenizer.decode(outputs[0], skip_special_tokens=True)})
    
    except Exception as e:
        logger.error(f"Error in /paraphrase: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Preload the model during server startup
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to preload model: {str(e)}")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
