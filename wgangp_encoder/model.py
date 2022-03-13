import torch

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 512, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.GELU(),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(1024, 2048, kernel_size=3, padding=0, stride=2),
            torch.nn.BatchNorm2d(2048),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(2048, 4096, kernel_size=3, padding=0, stride=2),
            torch.nn.BatchNorm2d(4096),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
        )

        self.range = 1024.0
    
    def forward(self, x):
        x = x / self.range
        x = self.block(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.block = torch.nn.Sequential(
            # nz will be the input to the first convolution
            torch.nn.ConvTranspose2d(
                4096, 2048, kernel_size=8, 
                stride=4, padding=0, bias=False),
            torch.nn.BatchNorm2d(2048),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                2048, 1024, kernel_size=6, 
                stride=3, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ConvTranspose2d(
                1024, 512, kernel_size=8,
                stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                512, 256, kernel_size=8, 
                stride=2, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                256, 1, kernel_size=3, 
                stride=1, padding=0, bias=False),
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.range = 1024.0
    
    def forward(self, x):
        x = self.block(x)
        x = self.sigmoid(x) * self.range
        return x

class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(3),
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
        )

        self.critic_out = torch.nn.Sequential(
            torch.nn.Linear(1024 * 2 * 2, 1),
        )

        self.range = 1024.0
        
    def forward(self, x):
        x = x / self.range
        x = self.block(x)
        x = x.view([-1, 1024*2*2])
        x = self.critic_out(x)
        return x