import model
import torch

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = model.ImageEncoder([1, 16, 32, 64, 128, 256, 512], kernel_size=3, stride=2)
        self.decoder = model.ImageDecoder128()

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.range = 1024.0

    def forward(self, image):
        x = image / self.range
        x = self.encoder(image)
        x = self.tanh(x)
        x = self.decoder(x)        

        x = self.sigmoid(x) * self.range

        return x