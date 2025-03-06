# Humanize-AI-Server

**Humanize AI** is a text rewriting and paraphrasing service that leverages advanced T5-based models from [Hugging Face](https://huggingface.co). The goal is to generate text that preserves meaning while making it sound more natural and less detectable by AI detectors. 

> **Note**: This project is still under active development and not all features are finalized.

---

## Project Structure

```
.
├── api
│   ├── index.py              # Main Flask server
│   ├── download_model.py     # Script to download and save T5 model locally
│   ├── models.txt            # List of model names or references
│   ├── requirements.txt      # Python dependencies
│   ├── vercel.json           # Vercel deployment configuration
│   ├── makefile              # (Optional) Make commands for building/deployment
│   └── ...
├── .env                      # Environment variables (HF_TOKEN, etc.)
└── .gitignore
```

- **index.py**: Contains your Flask application with endpoints (`/`, `/hello`, `/paraphrase`).
- **download_model.py**: Downloads the `t5-small` model and tokenizer to a local folder.
- **models.txt**: Lists models or references to be used or downloaded.
- **requirements.txt**: Lists Python dependencies (Flask, requests, python-dotenv, etc.).
- **vercel.json**: Configuration file if you choose to deploy on Vercel.

---

## Features

- **Paraphrasing**  
  Leverages a T5-based paraphrasing model (e.g., `humarin/chatgpt_paraphraser_on_T5_base` or `Vamsi/T5_Paraphrase_Paws`) via the Hugging Face Inference API.

- **Custom Models**  
  You can switch out the model URL in `index.py` (the `HF_API_URL`) to any other Hugging Face model endpoint.

- **Environment Variables**  
  Uses a `.env` file to store sensitive tokens (e.g., `HF_TOKEN`).

- **Logging**  
  Uses Python’s `logging` module to debug and monitor API behavior.

---

## Prerequisites

1. **Python 3.11+**  
2. **Hugging Face Account & Token**  
   - Generate a token from your Hugging Face profile settings.
   - Create a `.env` file and add `HF_TOKEN=your_huggingface_token_here`.

---

## Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/YOUR_USERNAME/Humanize-AI-Server.git
   cd Humanize-AI-Server/api
   ```

2. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**  
   - Create a `.env` file in the project root or `api` folder with:
     ```ini
     HF_TOKEN=your_huggingface_token_here
     ```
   - Make sure to keep this file secret or add it to your `.gitignore`.

5. **(Optional) Download a local T5 model**  
   - If you want to download and store a local T5 model (e.g., `t5-small`), run:
     ```bash
     python download_model.py
     ```
   - This will create a folder `./models/t5-small` with the tokenizer and model weights.  
   - *Note:* The current Flask code in `index.py` uses the Hugging Face Inference API, so downloading the model locally is optional.

---

## Running the Flask App

From inside the `api` directory, run:

```bash
python index.py
```

The Flask server will start on `http://127.0.0.1:5001` by default (or another port if you set the `PORT` environment variable).

### Endpoints

1. **Health Check**  
   - **GET /**  
   - **Example:** `curl http://127.0.0.1:5001/`  
   - **Response:**  
     ```
     Server operational!
     ```

2. **Hello Test**  
   - **GET /hello**  
   - **Example:** `curl http://127.0.0.1:5001/hello`  
   - **Response:**  
     ```
     Hello, Vercel deployment works!
     ```

3. **Paraphrase**  
   - **POST /paraphrase**  
   - **Request Body (JSON):**  
     ```json
     {
       "text": "Your text to paraphrase goes here."
     }
     ```
   - **Response (JSON):**  
     ```json
     {
       "paraphrased": "The generated paraphrased text.",
       "success": true
     }
     ```

---

## Deploying on Vercel

This project includes a **vercel.json** configuration for deploying a Python server.  
1. Install the [Vercel CLI](https://vercel.com/docs/cli).  
2. Run `vercel` in the project root and follow the prompts.  

*Note:* Python/Flask on Vercel typically uses the [Serverless Functions](https://vercel.com/docs/serverless-functions) approach. Make sure your `vercel.json` and project structure match Vercel’s Python requirements.

---

## Contributing

1. **Fork** the repository.  
2. **Create a new branch** for your feature or bug fix.  
3. **Commit** your changes with clear messages.  
4. **Submit a Pull Request** to the main branch.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For any questions or suggestions, please open an issue in the repository or contact [odukoyaabdullahi01@gmail.com](mailto:odukoyaabdullahi01@gmail.com).

---

### Important Note
This project is still under active development. The paraphrasing and “humanization” features may not be perfect, and the underlying model can be changed or improved at any time. Feel free to experiment with different Hugging Face models for better performance.
