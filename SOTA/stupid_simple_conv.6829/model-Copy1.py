import torch

class DeepConvChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.range = 1024.0
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(12, 512, kernel_size=3, padding=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.GELU(),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(1024, 2048, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(2048),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(2048, 4096, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(4096),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
        )
        
        self.decoder = torch.nn.Sequential(
            # nz will be the input to the first convolution
            torch.nn.ConvTranspose2d(
                4096, 2048, kernel_size=4, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(512),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                2048, 1024, kernel_size=4, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(256),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                1024, 512, kernel_size=4, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(
                512, 24, kernel_size=3, 
                stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(64),
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        

    def forward(self, x):
        x = x / self.range
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sigmoid(x) * self.range
        return x