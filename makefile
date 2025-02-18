uvicorn:
	uvicorn nlp_server:app --host 0.0.0.0 --port 8000 --reload

uvicorn-reload:
	uvicorn nlp_server:app --reload

# DP SK
start-server:
	python api/index.py

# Install dependencies
requirements:
	pip install -r requirements.txt

activate-venv:
	source .venv/bin/activate

create-venv-environment:
	python3 -m venv .venv

install-dependencies:
	pip install flask transformers huggingface_hub torch

# Start the server
# start-server:
# 	python models/openai_t5_server.py
