# Load a question answering dataset, load first 100
qa_dataset = load_dataset("squad", split="train[:100]")

# Examine question–answer pair
example = qa_dataset[0]
print("Context:", example["context"][:300] + "...")
print("Question:", example["question"])
print("Answer:", example["answers"]["text"][0])
print("Answer starts at position:", example["answers"]["answer_start"][0])
