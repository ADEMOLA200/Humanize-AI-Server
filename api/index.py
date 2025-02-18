import os
import logging
import signal
from contextlib import contextmanager
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from huggingface_hub import snapshot_download, login
import torch
from transformers.utils import logging as hf_logging

os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

hf_logging.set_verbosity_debug()

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment")

# Debug: Print first few characters of the token
logger.info(f"Hugging Face Token (first 5 chars): {HF_TOKEN[:5]}****")

app = Flask(__name__)

model = None
tokenizer = None

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

MODEL_DIR = "./models/t5-small"

def load_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        try:
            if os.path.exists(MODEL_DIR):
                logger.info("Loading model from local storage...")
            else:
                logger.info("Local model not found. Downloading from Hugging Face...")
                login(os.getenv("HF_TOKEN"))  # We authenticate with Hugging Face
                snapshot_download(repo_id="t5-small", local_dir=MODEL_DIR)

            tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

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

        return jsonify({
            'paraphrased': tokenizer.decode(outputs[0], skip_special_tokens=True),
            'success': True
        })

    except Exception as e:
        logger.error(f"Error in /paraphrase: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to preload model: {str(e)}")

    # Start the Flask server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
