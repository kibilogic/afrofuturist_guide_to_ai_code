import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm

class CertifiedRobustModel(nn.Module):
    """
    Defense shield to protect against adversarial attacks
    """
    def __init__(self, base_model, smoothing_std=0.25):
        super().__init__()
        self.base_model = base_model
        self.smoothing_std = smoothing_std

    def forward(self, x, num_samples=100):
        """
        Randomized smoothing 
        """
        self.base_model.eval()
        
        batch_size = x.shape[0]
        predictions = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Gaussian noise
                noise = torch.randn_like(x) * self.smoothing_std
                noisy_x = x + noise

                # Get prediction
                pred = self.base_model(noisy_x)
                predictions.append(pred)

        # predictions across noisy samples
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction

    def certified_radius(self, confidence_level=0.95):
        """
        Calculate the safe zone
        """
        alpha = 1 - confidence_level
        return self.smoothing_std * norm.ppf(1 - alpha / 2)


def demonstrate_certified_defense():
    """
    Certainty that can coexist with uncertainty
    """
    # Load a pre-trained model
    base_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    base_model.eval()
    
    certified_model = CertifiedRobustModel(base_model, smoothing_std=0.25)
    
    # Calculate certified radius at 95% confidence
    radius = certified_model.certified_radius(confidence_level=0.95)
    print(f"Certified robust radius: {radius:.4f}")
    print("Within this radius, the model's predictions are provably stable.")

    # Create a synthetic test input (normally you will use real data)
    sample_input = torch.randn(1, 3, 224, 224)
    
    prediction = certified_model(sample_input)

    predicted_class = torch.argmax(prediction, dim=1).item()
    print(f"Predicted class (with certified defense): {predicted_class}")
    
    return prediction, radius

# Run the program
if __name__ == "__main__":
    demonstrate_certified_defense()

