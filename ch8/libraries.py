from transformers import pipeline

# Create a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text from a prompt
prompt = "The great libraries of Timbuktu contained"
result = generator(prompt, max_new_tokens=256, pad_token_id=50256, num_return_sequences=1)
print(result[0]['generated_text'])
