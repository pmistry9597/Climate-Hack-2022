from .model import ImageEncoder, ImageDecoder128
import torch

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder([1, 128, 256, 512, 1024, 2048, 4096], kernel_size=3, stride=2)
        self.decoder = ImageDecoder128()

        self.tanh = torch.nn.Tanh()
#         self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()

        self.range = 1024.0

    def forward(self, image):
        x = image / self.range
        x = self.encoder(x)
        # activate here if u want, tho prob bad idea :(
        x = self.tanh(x) #maybe not so bad idea, enforces latent space to be between -1 and 1
        x = self.decoder(x)        

        x = self.sigmoid(x) * self.range

        return x