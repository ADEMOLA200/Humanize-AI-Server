from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define the model directory
model_dir = "./models/t5-small"

# Download and save the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.save_pretrained(model_dir)

# Download and save the model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.save_pretrained(model_dir)

print(f"Model and tokenizer saved to {model_dir}")
