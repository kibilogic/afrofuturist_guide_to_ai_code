from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import random

# Use model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create pipeline for text generation
law_helper = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# Define Legal References
legal_examples = [
    {
        "title": "Miranda Rights",
        "text": """
You have the right to remain silent. Anything you say can and will be used against you in a court of law.
You have the right to an attorney. If you cannot afford one, one will be provided for you.
"""
    },
    {
        "title": "What Happens in Traffic Court",
        "text": """
If you plead not guilty to a traffic ticket, you must appear in traffic court.
A judge will hear your case and you may present evidence or witnesses.
The judge will then decide whether you are guilty and assign a fine or penalty.
"""
    },
    {
        "title": "Your Right to Remain Silent",
        "text": """
The Fifth Amendment of the U.S. Constitution protects you from having to answer questions that might incriminate you.
You have the right to remain silent and not respond to police questions without a lawyer present.
"""
    }
]

# Randomly Select One
selected = random.choice(legal_examples)
print(f"Selected Topic: {selected['title']}")


# Use LLM to Generate Simplified Explanation
prompt = f"Rephrase this legal information so it's easy for a high school student to understand:\n\n{selected['text']}"
response = law_helper(prompt, max_new_tokens=200)[0]['generated_text']

print("AI Rephrased Answer:\n")
print(response)

