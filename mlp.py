import torch
import torch.nn as nn
import torch.optim as optim

class mlp(nn.Module):
    def __init__(self, latent_dim, hidden_dim1, hidden_dim2):
        super(mlp, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.fc1 = nn.Linear(latent_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim1, 1)
        #self.fc3 = nn.Linear(hidden_dim2, 1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        out = self.fc3(out)
        return out