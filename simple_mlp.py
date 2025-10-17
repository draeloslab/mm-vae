import torch 
import torch.nn as nn
import torch.optim as optim 

class simple_mlp(nn.Module):
    def __init__(self, latent_dim=5):
        super(simple_mlp, self).__init__()
        self.latent_dim = latent_dim
        #self.fc1 = nn.Linear(latent_dim + 1, 20)
        self.fc1 = nn.Linear(latent_dim, 20)
        self.tanh1 = nn.Tanh()
        #self.dropout = nn.Dropout(p=0.5)
        #self.relu = nn.ReLU()]
        self.fc2 = nn.Linear(20, 1)
        #self.tanh2 = nn.Tanh()
        #self.tanh = nn.Tanh()
        #self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.fc1(x)
        #out = self.relu(out)
        out = self.tanh1(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        #out = self.tanh2(out)
        #out = self.fc3(out)  
        return out 
    


