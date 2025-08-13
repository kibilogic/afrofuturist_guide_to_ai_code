import torch
import torch.nn as nn
import random

class DiverseEnsemble(nn.Module):
    """
    Create system where multiple models evaluate the same input
    """
    def __init__(self, models, diversity_regularization=0.1):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.diversity_reg = diversity_regularization
        
    def forward(self, x, training=False):
        outputs = []
        
        for model in self.models:
            if training:
                # Add slight randomness 
                noise = torch.randn_like(x) * 0.01
                output = model(x + noise)
            else:
                output = model(x)
            outputs.append(output)
        
        # Stack outputs and compute ensemble prediction
        stacked_outputs = torch.stack(outputs)
        ensemble_output = stacked_outputs.mean(dim=0)
        
        if training:
            # Encourage diversity 
            diversity_loss = self.compute_diversity_loss(stacked_outputs)
            return ensemble_output, diversity_loss
        else:
            return ensemble_output
    
    def compute_diversity_loss(self, stacked_outputs):
        """
        Encourage models to disagree on adversarial examples
        while agreeing on clean examples
        """
        # Compute pairwise disagreement
        num_models = len(self.models)
        diversity_loss = 0
        
        for i in range(num_models):
            for j in range(i+1, num_models):
                # KL divergence between model outputs
                p = torch.softmax(stacked_outputs[i], dim=1)
                q = torch.softmax(stacked_outputs[j], dim=1)
                kl_div = torch.sum(p * torch.log(p / (q + 1e-8)), dim=1).mean()
                diversity_loss -= kl_div  # Negative for diversity
        
        return self.diversity_reg * diversity_loss

def adaptive_ensemble_decision(ensemble, x, uncertainty_threshold=0.1):
    """
    The ensemble adapts its decision-making process based on
    the level of uncertainty in the situation
    """
    with torch.no_grad():
        # Get predictions from all models
        individual_outputs = []
        for model in ensemble.models:
            output = torch.softmax(model(x), dim=1)
            individual_outputs.append(output)
        
        stacked_outputs = torch.stack(individual_outputs)
        
        # Measure disagreement among models
        mean_output = stacked_outputs.mean(dim=0)
        disagreement = torch.var(stacked_outputs, dim=0).sum(dim=1)
        
        # High disagreement suggests potential adversarial input
        high_uncertainty = disagreement > uncertainty_threshold
        
        # For high uncertainty cases, use more conservative voting
        if high_uncertainty.any():
            # Use median instead of mean for uncertain cases
            median_output = torch.median(stacked_outputs, dim=0)[0]
            final_output = torch.where(high_uncertainty.unsqueeze(1),
                                       median_output,
                                       mean_output)
        else:
            final_output = mean_output
            
        return final_output, disagreement


# Run program
if __name__ == "__main__":
    # Simple models for testing
    class SimpleNet(nn.Module):
        def __init__(self, input_size=10, num_classes=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    # Generate dummy ensemble
    num_models = 3
    ensemble_models = [SimpleNet() for _ in range(num_models)]
    ensemble = DiverseEnsemble(ensemble_models)

    # Dummy input batch of size 5, with 10 features each
    x = torch.randn(5, 10)

    # Training mode forward (returns output + diversity loss)
    output, diversity_loss = ensemble(x, training=True)
    print("Training Output:\n", output)
    print("Diversity Loss:", diversity_loss.item())

    # Inference mode with adaptive decision
    final_output, disagreement = adaptive_ensemble_decision(ensemble, x)
    print("Final Output:\n", final_output)
    print("Disagreement Score:", disagreement)

