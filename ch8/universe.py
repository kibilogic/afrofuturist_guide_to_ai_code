# Install packages if necessary
# !pip install transformers

from transformers import pipeline

# BERT - Masked Language Modeling
bert_fill = pipeline("fill-mask", model="bert-base-uncased")
bert_input = "The [MASK] was filled with stars and galaxies."
bert_output = bert_fill(bert_input)

print("BERT's Interpretation:")
print(f"Input: {bert_input}")
for pred in bert_output[:3]:
    print(f"- {pred['token_str']} (score: {pred['score']:.4f})")

# GPT - Text Generation
gpt_generate = pipeline("text-generation", model="gpt2")
gpt_input = "The universe is a vast and mysterious place"
gpt_output = gpt_generate(gpt_input, max_new_tokens=30, do_sample=True, temperature=0.7)

print("\nGPT's Continuation:")
print(f"Prompt: {gpt_input}")
print(f"Output: {gpt_output[0]['generated_text']}")
