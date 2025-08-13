import matplotlib.pyplot as plt
import networkx as nx

# Perceptron class 
class SmokeSignalPerceptron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def signal(self, inputs):
        self.inputs = inputs
        self.weighted_sum = sum(w * x for w, x in zip(self.weights, inputs))
        self.output = 1 if self.weighted_sum >= self.threshold else 0
        return self.output

# If raised the flag (1, 1), alert is (output 1)
perceptron = SmokeSignalPerceptron(weights=[1, 1], threshold=2)
output = perceptron.signal([1, 1])  

G = nx.DiGraph()

# Nodes: Inputs, Weights, Sum, Threshold, Output
G.add_node("x1", pos=(0, 2))
G.add_node("x2", pos=(0, 0))
G.add_node("sum", pos=(2, 1))
G.add_node("threshold", pos=(4, 1))
G.add_node("output", pos=(6, 1))

# Edges: input -> sum (with weights), sum -> threshold, threshold -> output
G.add_edge("x1", "sum", label=f"w1={perceptron.weights[0]}")
G.add_edge("x2", "sum", label=f"w2={perceptron.weights[1]}")
G.add_edge("sum", "threshold", label=f"∑={perceptron.weighted_sum}")
G.add_edge("threshold", "output", label=f">= {perceptron.threshold}?")

# Positions for layout
pos = nx.get_node_attributes(G, 'pos')

# Nodes
plt.figure(figsize=(10, 5))

nx.draw_networkx_nodes(
    G,
    pos,
    node_color='white',       # White fill
    edgecolors='black',       # Optional: black border to keep them visible
    linewidths=1.5,           # Slightly thicker border
    node_size=1500
)

nx.draw_networkx_labels(G, pos)

# Edges and edge labels
nx.draw_networkx_edges(G, pos, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title(f"SmokeSignalPerceptron Output: {perceptron.output}", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

