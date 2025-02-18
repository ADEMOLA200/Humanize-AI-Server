import os
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv

os.environ["TORCH_DYNAMO_DISABLE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# we will use a model fine-tuned for paraphrasing from huggingface
HF_API_URL = "https://api-inference.huggingface.co/models/humarin/chatgpt_paraphraser_on_T5_base"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment")

logger.info(f"Hugging Face Token (first 5 chars): {HF_TOKEN[:5]}****")

app = Flask(__name__)

@app.route('/')
def health_check():
    return "Server operational!"

@app.route('/hello')
def hello():
    return "Hello, Vercel deployment works!"

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": text
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        api_response = response.json()
        logger.debug(f"Hugging Face API Response: {api_response}")

        generated_text = api_response[0].get("generated_text")
        if generated_text:
            return jsonify({'paraphrased': generated_text, 'success': True})

        return jsonify({"error": "Unexpected response format", "response": api_response}), 500

    except Exception as e:
        logger.error(f"Error in /paraphrase: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))
