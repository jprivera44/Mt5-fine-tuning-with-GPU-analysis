# Import PyTorch
import torch
import torch.nn as nn

# Define a simple two-layer feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the network
    net = SimpleNet(500, 500, 10).to(device)

    # Create a random tensor to represent input data
    x = torch.randn(128, 500).to(device)

    # Run the network on the input data
    output = net(x)

if __name__ == "__main__":
    main()