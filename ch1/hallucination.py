from transformers import pipeline


generator = pipeline("text-generation", model="gpt2")

# Asks for a historical explanation on a nonexistent role in the moon landing
prompt = (
    "Explain the history of the fictional country of Wakandia and its contributions to the moon landing."
)

# Response will almost certainly include made-up facts, names, and historical events, presented confidently
output = generator(prompt, max_new_tokens=100, pad_token_id=50256,do_sample=True, temperature=0.9)

print("AI Output:\n", output[0]['generated_text'])


