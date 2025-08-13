# RLHF (Reinforcement Learning from Human Feedback)

# Simulation...train model using human feedback 

print("\n--- Teaching the Model Values (Conceptual RLHF) ---")

# Step 1: Human Feedback Collection
print("\n1. Collecting Human Feedback:")
print("   People compare model answers and say which one is better.")
print("   Example: 'Answer A is better than Answer B for this question.'")

human_feedback = [
    {
        "prompt": "Tell me about climate change.",
        "response_A": "Climate change is a natural phenomenon.",
        "response_B": ("Climate change is mostly caused by humans "
                       "and needs urgent action."),
        "preferred": "response_B"
    },
    {
        "prompt": "How can I protect my online privacy?",
        "response_A": ("Use strong passwords, turn on two-factor login, "
                       "and avoid clicking on strange links."),
        "response_B": "Privacy isn’t important. Just use your devices.",
        "preferred": "response_A"
    }
]

print("   Sample preference:",
      f"{human_feedback[0]['preferred']} was preferred for prompt 1.")

# Reward Model – Model learns what humans prefer
print("\n2. Training the Reward Model:")
print("   The Reward Model scores answers based on how good they are.")
print("   It learns from human feedback.")

import torch

class SimpleRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fake_layer = torch.nn.Linear(10, 1)  

    def forward(self, prompt_vector, response_vector):
        # returns a score based on how long is the response 
        score = len(response_vector) * 0.1
        return torch.tensor(score, dtype=torch.float32)

reward_model = SimpleRewardModel()
print("   Reward Model is ready.")

# Reinforcement Learning (Making the model better)
print("\n3. Reinforcement Learning (Concept):")
print("   The model generates an answer, gets a reward score,")
print("   and learns to do better next time.")

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)

class FakeLLMAgent:
    def __init__(self, model, tokenizer, reward_model):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model

    def generate_response(self, prompt):
        tokens = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **tokens, max_new_tokens=20,
            do_sample=True, top_k=50, temperature=0.7
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def get_reward(self, prompt, response):
        prompt_vec = [1] * 5
        response_vec = [1] * len(response.split())
        reward = self.reward_model(
            torch.tensor(prompt_vec),
            torch.tensor(response_vec)
        ).item()

        print(f"    - Prompt: {prompt}")
        print(f"    - Response: {response}")
        print(f"    - Reward Score: {reward:.2f}")
        print("    - (In real training, model would update here!)")
        return reward


# Simulate responses
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

class FakeGenerativeModel(torch.nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

    def generate(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        prompt_text = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )

        if "climate change" in prompt_text.lower():
            return self.tokenizer.encode(
                "Climate change is caused by humans and needs urgent action.",
                return_tensors="pt"
            )
        elif "online privacy" in prompt_text.lower():
            return self.tokenizer.encode(
                "Use strong passwords and enable two-factor login.",
                return_tensors="pt"
            )
        return self.tokenizer.encode(
            "This is a model response.",
            return_tensors="pt"
        )

fake_model = FakeGenerativeModel(base_model, tokenizer)
agent = FakeLLMAgent(fake_model, tokenizer, reward_model)

# RLHF steps (2 rounds)
print("\nRunning RLHF simulation...\n")
for i in range(2):
    print(f"--- RLHF Round {i+1} ---")
    for sample in human_feedback:
        prompt = sample["prompt"]
        response = agent.generate_response(prompt)
        agent.get_reward(prompt, response)

print("\nModel training complete (conceptual).")
print("   The model now gives more helpful and safer answers.")

