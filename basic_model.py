# %%
import torch

# %%
class BasicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0 = torch.nn.Linear(in_features=12*128*128, out_features=1024)
        self.layer1 = torch.nn.Linear(in_features=1024, out_features=1024)
        self.layer2 = torch.nn.Linear(in_features=1024, out_features=24*64*64)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, features):
        x = features.view([-1, 12*128*128])

        x = self.relu(self.layer0(x))
        x = self.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))

        return x.view([-1, 24, 64, 64]) * 1024.0


