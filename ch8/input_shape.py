from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer automatically
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Process text
text = "Traditional knowledge systems hold valuable insights"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print("Input shape:", inputs['input_ids'].shape)
print("Output shape:", outputs.last_hidden_state.shape)
