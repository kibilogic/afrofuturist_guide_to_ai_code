import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Neural network 
class KenteWeaverNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

def visualize_kente_loom_architecture(model):
    """
    Visualize neural network as a kente loom with textile-inspired styling
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define positions for each layer
    input_positions = [(1, 3), (1, 1)]  # 2 input nodes
    hidden_positions = [(4, i) for i in np.linspace(0.5, 3.5, 6)]  # 6 hidden nodes
    output_positions = [(7, 2)]  # 1 output node
    
    # Kente-inspired colors
    colors = {
        'input': '#FFD700',      # Gold for input threads
        'hidden': '#FF6B35',     # Orange-red for weaving points
        'output': '#8B4513',     # Brown for final motif
        'strong_pos': '#B22222', # Dark red for strong positive connections
        'weak_neg': '#696969'    # Gray for weak/negative connections
    }
    
    # Get weights 
    with torch.no_grad():
        weights_layer1 = model.fc1.weight.data.numpy()  
        weights_layer2 = model.fc2.weight.data.numpy()  
    
    # Draw connections 
    def draw_connections(start_positions, end_positions, weights, layer_name):
        connections_drawn = 0
        print(f"\n{layer_name} - Weight matrix shape: {weights.shape}")
        print(f"Drawing connections from {len(start_positions)} start nodes to {len(end_positions)} end nodes")
        
        for i, start_pos in enumerate(start_positions):
            for j, end_pos in enumerate(end_positions):
                weight = weights[j, i]  # weights are stored as [output, input]
                
                if weight > 0:
                    color = colors['strong_pos']
                    alpha = max(min(abs(weight) * 0.8, 1.0), 0.3)  
                    linewidth = max(abs(weight) * 2, 1.0)  
                else:
                    color = colors['weak_neg']
                    alpha = max(min(abs(weight) * 0.8, 0.9), 0.3)  
                    linewidth = max(abs(weight) * 2, 0.8)  
                
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       color=color, alpha=alpha, linewidth=linewidth, zorder=1)
                connections_drawn += 1
                
        print(f"Total connections drawn: {connections_drawn}")
        print(f"Expected connections: {len(start_positions) * len(end_positions)}")
    
    print("Debugging Neural Network Connections:")
    print(f"Layer 1 weights shape: {weights_layer1.shape}")
    print(f"Layer 2 weights shape: {weights_layer2.shape}")
    print(f"Layer 1 weight range: {weights_layer1.min():.3f} to {weights_layer1.max():.3f}")
    print(f"Layer 2 weight range: {weights_layer2.min():.3f} to {weights_layer2.max():.3f}")
    
    # Draw connections
    draw_connections(input_positions, hidden_positions, weights_layer1, "Layer 1 (Input->Hidden)")
    draw_connections(hidden_positions, output_positions, weights_layer2, "Layer 2 (Hidden->Output)")
    
    # Input nodes (threads)
    for i, pos in enumerate(input_positions):
        circle = plt.Circle(pos, 0.15, color=colors['input'], ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0]-0.4, pos[1], f'Input {i+1}', fontsize=10, ha='right', va='center', weight='bold')
    
    # Hidden nodes (weaving points)
    for i, pos in enumerate(hidden_positions):
        circle = plt.Circle(pos, 0.12, color=colors['hidden'], ec='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1]+0.25, f'H{i+1}', fontsize=9, ha='center', va='bottom', weight='bold')
    
    # Output node (final motif)
    for i, pos in enumerate(output_positions):
        circle = plt.Circle(pos, 0.18, color=colors['output'], ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[0]+0.4, pos[1], 'Output', fontsize=10, ha='left', va='center', weight='bold')
    
    ax.text(1, 4, 'Input Threads', fontsize=12, ha='center', weight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['input'], alpha=0.7))
    
    ax.text(4, 4.2, 'Hidden Weaving Points', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['hidden'], alpha=0.7))
    
    ax.text(7, 3, 'Output Motif', fontsize=12, ha='center', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['output'], alpha=0.7))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['input'], lw=8, label='Input Threads'),
        plt.Line2D([0], [0], color=colors['hidden'], lw=8, label='Hidden Weaving Points'),
        plt.Line2D([0], [0], color=colors['output'], lw=8, label='Output Motif'),
        plt.Line2D([0], [0], color=colors['strong_pos'], lw=3, label='Strong Positive Connection'),
        plt.Line2D([0], [0], color=colors['weak_neg'], lw=2, label='Weak/Negative Connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.9)
    
    # Styling
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Kente-Inspired Neural Loom Architecture', fontsize=16, weight='bold', pad=20)
    
    # Add grid pattern to mimic textile 
    for x in np.arange(0.5, 8, 0.5):
        ax.axvline(x, color='lightgray', alpha=0.3, linewidth=0.5, zorder=0)
    for y in np.arange(0.5, 4.5, 0.5):
        ax.axhline(y, color='lightgray', alpha=0.3, linewidth=0.5, zorder=0)
    
    plt.tight_layout()
    plt.show()

def demonstrate_information_flow(model, sample_input):
    """
    Show how information flows through the network layers
    """
    with torch.no_grad():
        # Forward pass with intermediate outputs
        layer1_output = model.relu(model.fc1(sample_input))
        final_output = model.sigmoid(model.fc2(layer1_output))
        
        print("Information Flow Through the Kente Loom:")
        print(f"Input Threads: {sample_input.numpy()}")
        print(f"Hidden Weaving Pattern: {layer1_output.numpy().round(3)}")
        print(f"Final Motif: {final_output.numpy().round(3)}")
        
        # Visualize activation levels
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Input visualization
        ax1.bar(['Thread 1', 'Thread 2'], sample_input.numpy(), color='#FFD700', alpha=0.8)
        ax1.set_title('Input Threads', fontsize=12, weight='bold')
        ax1.set_ylabel('Activation Level')
        
        # Hidden layer visualization
        hidden_activations = layer1_output.numpy()
        ax2.bar([f'H{i+1}' for i in range(6)], hidden_activations, color='#FF6B35', alpha=0.8)
        ax2.set_title('Hidden Weaving Points', fontsize=12, weight='bold')
        ax2.set_ylabel('Activation Level')
        ax2.tick_params(axis='x', rotation=45)
        
        # Output visualization
        ax3.bar(['Final Motif'], final_output.numpy(), color='#8B4513', alpha=0.8)
        ax3.set_title('Output Motif', fontsize=12, weight='bold')
        ax3.set_ylabel('Prediction Probability')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Create and train the model 
    from sklearn.datasets import make_classification
    
    # Create dataset
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1, n_classes=2, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # Create and train model
    model = KenteWeaverNet()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    for epoch in range(100):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Visualizing the Kente Neural Loom Architecture...")
    visualize_kente_loom_architecture(model)
    
    print("\nDemonstrating Information Flow...")
    sample_input = torch.tensor([1.5, -0.5], dtype=torch.float32)
    demonstrate_information_flow(model, sample_input)

    