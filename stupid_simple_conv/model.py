import torch

class DeepConvChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.garbo = torch.nn.Sequential(
            torch.nn.Conv2d(12, 1024, kernel_size=3, padding=2),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
            torch.nn.Conv2d(1024, 4096, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(4096),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(4096, 8192, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(8192),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8192, 16384, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(16384),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.garbo(x)