# Install required packages
!pip install transformers
from transformers import pipeline

# Load GPT-2 AI model
chatbot = pipeline("text-generation", model="gpt2")

# Predefined responses
command_responses = {
    "play music": "Sure, playing your favorite playlist now!",
    "what’s the weather": "It looks sunny with a chance of creativity!",
    "set alarm": "Alarm set for 7:00 AM tomorrow.",
    "tell me a proverb": "The child who is not embraced by the village will burn it down to feel its warmth."
}

# Chatbot logic function
def smart_chatbot(user_input):
    lower_input = user_input.lower()
    for command in command_responses:
        if command in lower_input:
            return command_responses[command]

    # If unknown input, generate response with GPT-2
    generated = chatbot(
        f"User: {user_input}\nAI:",
        max_new_tokens=250,
        pad_token_id=50256,
        do_sample=True
    )[0]['generated_text']

    return generated.split("AI:")[-1].strip()

response = smart_chatbot("play music")
print("Bot:", response)
