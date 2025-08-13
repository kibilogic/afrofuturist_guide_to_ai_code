from transformers import pipeline

# Load a summarization model 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example
text = (
    "Long ago, in a village where drought was common, a tortoise convinced the "
    "animals that he could bring rain if they each brought him a gift. Every animal "
    "gave something: the elephant offered fruit, the lion gave meat, and the bird "
    "brought seeds. The tortoise collected the gifts but never performed the ritual. "
    "Eventually, the sky rumbled, not with rain—but with thunder of the angry animals "
    "who had been deceived."
)

# Run summarization
summary = summarizer(text, max_length=60, min_length=20, do_sample=False)

print("Original Story:\n", text)
print("\nModel-Generated Summary:\n", summary[0]['summary_text'])

