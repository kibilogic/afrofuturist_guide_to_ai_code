import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
#660

def generate_adversarial_example(model, data, target, epsilon=0.03):
    """
    reveal hidden vulnerabilities, prediction—simulating attack
    """
    data_grad = Variable(data, requires_grad=True)

    output = model(data_grad)
    loss = nn.CrossEntropyLoss()(output, target)

    model.zero_grad()
    loss.backward()

    sign_data_grad = data_grad.grad.data.sign()

    perturbed_data = data + epsilon * sign_data_grad
    
    return torch.clamp(perturbed_data, 0, 1)

def adversarial_training_step(model, clean_data, clean_targets, optimizer):
    """
    Train the model 
    """
    model.train()
    
    # Generate adversarial examples
    adv_data = generate_adversarial_example(model, clean_data, clean_targets)
    
    # Train on both clean and adversarial 
    optimizer.zero_grad()

    clean_output = model(clean_data)
    clean_loss = nn.CrossEntropyLoss()(clean_output, clean_targets)
    
    # Adversarial examples
    adv_output = model(adv_data)
    adv_loss = nn.CrossEntropyLoss()(adv_output, clean_targets)
    
    # Combined loss 
    total_loss = (clean_loss + adv_loss) / 2
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

class SimpleCNN(nn.Module):
    def __init__(self):
        """
        Convolutional Neural Network to demonstrate adversarial training
        """
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)  
        
        # Dynamically determine the flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            dummy_output = torch.relu(self.conv(dummy_input))
            self.flatten_size = dummy_output.view(1, -1).size(1)  
        
        self.fc = nn.Linear(self.flatten_size, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Run program
if __name__ == "__main__":
    # Create model
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulate one batch of training  
    clean_data = torch.rand(16, 1, 28, 28)  
    clean_targets = torch.randint(0, 10, (16,))  

    clean_data = clean_data
    clean_targets = clean_targets

    loss = adversarial_training_step(model, clean_data, clean_targets, optimizer)
    print(f"Adversarial training loss: {loss:.4f}")

    
