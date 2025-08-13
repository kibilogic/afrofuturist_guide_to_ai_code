# Install Packages
!pip install -q crewai transformers accelerate bitsandbytes langchain langchain_community langchain-huggingface

# Translation Models
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianTokenizer, MarianMTModel

from google.colab import userdata
import os

# Requires Huggingface API Key, Stored in Google Colab
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

def wolof_to_french(text):
    tokenizer = AutoTokenizer.from_pretrained("Lahad/nllb200-francais-wolof")
    model = AutoModelForSeq2SeqLM.from_pretrained("Lahad/nllb200-francais-wolof")
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def french_to_english(text):
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

from crewai import Agent, Task, Crew, LLM
import os

# Set up LiteLLM using HuggingFace supported Inference Provider
llm = LLM(
    provider="huggingface",
    api_key=os.environ["HF_TOKEN"],
    model="huggingface/together/deepseek-ai/DeepSeek-R1"
)

cultural_expert = Agent(
    role="Cultural Expert",
    goal="Explain the cultural relevance of a translation",
    backstory="You are a linguist specializing in African oral traditions and culture.",
    llm=llm
)

bias_checker = Agent(
    role="Bias Checker",
    goal="Detect possible misrepresentations or cultural bias in translations",
    backstory="You ensure that translations remain fair and accurate to original meanings.",
    llm=llm
)

synthesizer = Agent(
    role="Final Synthesizer",
    goal="Craft a final English version that respects culture and clarity",
    backstory="You integrate cultural insight and accuracy to produce the best version.",
    llm=llm
)

# Translation from Wolof -> French -> English
def run_pipeline(wolof_input):
    print("Original Wolof:", wolof_input)
    french = wolof_to_french(wolof_input)
    english = french_to_english(french)
    print("French:", french)
    print("English:", english)

    t1 = Task(
        description=f"Analyze the cultural meaning of the Wolof proverb '{wolof_input}' and its English translation '{english}'",
        agent=cultural_expert,
        expected_output="A paragraph explaining the cultural context of the proverb."
    )

    t2 = Task(
        description=f"Check for translation bias or loss of meaning in the English phrase '{english}'",
        agent=bias_checker,
        expected_output="A bullet list or paragraph highlighting any bias or cultural mismatch."
    )

    t3 = Task(
        description="Create a final English version that retains cultural meaning and clarity, based on previous analysis.",
        agent=synthesizer,
        context=[t1, t2],
        expected_output="A refined translation that reflects cultural nuance and avoids bias."
    )

    crew = Crew(agents=[cultural_expert, bias_checker, synthesizer], tasks=[t1, t2, t3], verbose=True)
    result = crew.kickoff()
    print("Final Version:\n", result)

# English translation "Whoever wants honey must brave the bees"
run_pipeline("Ku bëgg lem, ñeme yamb")


