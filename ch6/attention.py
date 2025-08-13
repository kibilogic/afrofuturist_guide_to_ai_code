import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta

class AttentionMapper:
    def __init__(self, participants):
        """
        Initialize the attention mapper with a list of participants
        """
        self.participants = participants
        self.n_participants = len(participants)
        self.attention_matrix = np.zeros((self.n_participants, self.n_participants))
        self.temporal_data = []
        self.participant_to_index = {name: i for i, name in enumerate(participants)}
        
    def record_attention(self, observer, target, timestamp=None, weight=1.0):
        """
        Record an attention event between two participants
        """
        if observer not in self.participant_to_index or target not in self.participant_to_index:
            print(f"Warning: Unknown participant '{observer}' or '{target}'")
            return
            
        obs_idx = self.participant_to_index[observer]
        target_idx = self.participant_to_index[target]
        self.attention_matrix[obs_idx][target_idx] += weight
        
        if timestamp is None:
            timestamp = datetime.now()
        self.temporal_data.append({
            'timestamp': timestamp,
            'observer': observer,
            'target': target,
            'weight': weight
        })
    
    def calculate_attention_probabilities(self):
        """
        Convert raw attention counts to probability distributions
        """
        prob_matrix = np.zeros_like(self.attention_matrix)
        for i in range(self.n_participants):
            row_sum = np.sum(self.attention_matrix[i])
            if row_sum > 0:
                prob_matrix[i] = self.attention_matrix[i] / row_sum
        return prob_matrix
    
    def get_attention_stats(self):
        """
        Calculate various attention statistics
        """
        prob_matrix = self.calculate_attention_probabilities()
        stats = {}

        incoming_attention = np.sum(self.attention_matrix, axis=0)
        stats['most_attention_received'] = self.participants[np.argmax(incoming_attention)]

        outgoing_attention = np.sum(self.attention_matrix, axis=1)
        stats['most_attention_given'] = self.participants[np.argmax(outgoing_attention)]

        attention_entropy = []
        for i in range(self.n_participants):
            probs = prob_matrix[i]
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
            attention_entropy.append(entropy)
        stats['attention_entropy'] = dict(zip(self.participants, attention_entropy))
        return stats
    
    def plot_attention_heatmap(self, figsize=(10, 8)):
        """
        Create a grayscale heatmap of attention patterns
        """
        prob_matrix = self.calculate_attention_probabilities()
        plt.figure(figsize=figsize)
        sns.heatmap(prob_matrix, 
                    xticklabels=self.participants, 
                    yticklabels=self.participants,
                    annot=True, 
                    fmt='.2f', 
                    cmap='Greys',
                    cbar_kws={'label': 'Attention Probability'})
        plt.title('Attention Pattern Heatmap')
        plt.xlabel('Target (Who Receives Attention)')
        plt.ylabel('Observer (Who Gives Attention)')
        plt.tight_layout()
        plt.show()

def simulate_meeting_attention():
    """
    Simulate attention patterns in a typical meeting scenario
    """
    participants = ["Alice (Leader)", "Bob", "Charlie", "Diana", "Eve"]
    mapper = AttentionMapper(participants)
    start_time = datetime.now()
    
    print("Simulating meeting attention patterns...")
    for i in range(5):
        for participant in ["Bob", "Charlie", "Diana", "Eve"]:
            mapper.record_attention(participant, "Alice (Leader)", start_time + timedelta(seconds=i*10))

    for i in range(20):
        if i % 4 == 0:
            mapper.record_attention("Alice (Leader)", "Bob", start_time + timedelta(seconds=50 + i*5))
            mapper.record_attention("Bob", "Alice (Leader)", start_time + timedelta(seconds=52 + i*5))
        if i % 3 == 0:
            mapper.record_attention("Charlie", "Diana", start_time + timedelta(seconds=55 + i*5))
            mapper.record_attention("Diana", "Charlie", start_time + timedelta(seconds=57 + i*5))
        if i % 6 == 0:
            mapper.record_attention("Eve", "Alice (Leader)", start_time + timedelta(seconds=60 + i*5))
            mapper.record_attention("Alice (Leader)", "Eve", start_time + timedelta(seconds=62 + i*5))
    return mapper

if __name__ == "__main__":
    attention_mapper = simulate_meeting_attention()

    print("\n=== Attention Statistics ===")
    stats = attention_mapper.get_attention_stats()
    print(f"Most attention received: {stats['most_attention_received']}")
    print(f"Most attention given: {stats['most_attention_given']}")

    attention_mapper.plot_attention_heatmap()

    print("\n=== Attention Probability Matrix ===")
    prob_matrix = attention_mapper.calculate_attention_probabilities()
    df = pd.DataFrame(prob_matrix, 
                      index=attention_mapper.participants, 
                      columns=attention_mapper.participants)
    print(df.round(3))


