import numpy as np
import matplotlib.pyplot as plt

# Simulated prior: 30% belief that public support is strong
prior = 0.3

# Simulated news signals (1 = strong support, 0 = weak/negative)
news_signals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]

# Likelihoods: P(news|belief is true) and P(news|belief is false)
p_signal_given_true = 0.8
p_signal_given_false = 0.3

beliefs = [prior]

# Bayesian update over each signal
for signal in news_signals:
    prior = beliefs[-1]
    
    if signal == 1:
        numerator = p_signal_given_true * prior
        denominator = numerator + p_signal_given_false * (1 - prior)
    else:
        numerator = (1 - p_signal_given_true) * prior
        denominator = numerator + (1 - p_signal_given_false) * (1 - prior)
    
    updated_belief = numerator / denominator
    beliefs.append(updated_belief)

# Plot belief updates
plt.figure(figsize=(10, 5))
plt.plot(range(len(beliefs)), beliefs, marker='o', color='indigo')
plt.title("Belief Updating Over News Events")
plt.xlabel("Day")
plt.ylabel("Public Support")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

