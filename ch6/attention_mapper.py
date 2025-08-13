import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
data = {
    "Language": ["English", "French", "Japanese", "Turkish", "Spanish", "Quechua"],
    "Example": [
        "He might come",
        "Il se peut qu’il vienne",
        "Kare wa kuru kamoshirenai",
        "Gelmiş",
        "Es posible que venga",
        "Chaymi hamunmi"
    ],
    "Translation": [
        "He might come",
        "It is possible that he comes",
        "He might come",
        "He apparently came",
        "It's possible he comes",
        "He is coming (I saw it)"
    ],
    "Certainty_Level": [
        "Low",
        "Low",
        "Low",
        "Medium",
        "Low",
        "High"
    ],
    "Strategy": [
        "Modal Verb",
        "Subjunctive",
        "Epistemic Particle",
        "Evidential Marker",
        "Subjunctive",
        "Evidential Marker"
    ],
    "Category": [
        "Modal System",
        "Subjunctive Mood",
        "Sentence-Final Particle",
        "Morphological Evidentiality",
        "Subjunctive Mood",
        "Evidential System"
    ]
}


df = pd.DataFrame(data)

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cross-Linguistic Strategies for Encoding Uncertainty", fontsize=14, fontweight='bold')

# Strategy Count ---
strategy_counts = df["Strategy"].value_counts()
sns.barplot(x=strategy_counts.values, y=strategy_counts.index, ax=axes[0], palette="gray")
axes[0].set_title("Frequency of Strategies")
axes[0].set_xlabel("Number of Languages")
axes[0].set_ylabel("Strategy")

# Certainty Level by Language ---
certainty_order = ["Low", "Medium", "High"]
sns.countplot(data=df, y="Language", hue="Certainty_Level", ax=axes[1],
              palette="gray", hue_order=certainty_order)
axes[1].set_title("Certainty Levels Across Languages")
axes[1].set_xlabel("Number of Examples")
axes[1].legend(title="Certainty", loc="upper right")

plt.tight_layout()
plt.show()

