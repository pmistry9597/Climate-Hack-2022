import torch

class ImageEncoder(torch.nn.Module):
    def __init__(self, channel_list, kernel_size, stride=1):
        super().__init__()

        convLayers = []
        for i in range(len(channel_list)-1):
            conv = torch.nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size=kernel_size, padding=kernel_size-1, stride=stride)
            convLayers.append(conv)
            act = torch.nn.GELU()
            convLayers.append(act)
        self.block = torch.nn.Sequential(*convLayers)

    def forward(self, x):
        x = x.view([-1, 1, 128, 128])
        return self.block(x)

class ImageDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            # nz will be the input to the first convolution
            torch.nn.ConvTranspose2d(
                512, 256, kernel_size=5, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                256, 128, kernel_size=5, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=6, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                64, 1, kernel_size=6, 
                stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return self.block(x)
        #return self.main(x)'

class PWFF(torch.nn.Module):
    def __init__(self, in_chan, inner_chan, out_chan=None):
        super().__init__()
        if out_chan == None:
            out_chan = in_chan
        self.lin_in = torch.nn.Linear(in_chan, inner_chan)
        self.gelu = torch.nn.GELU()
        self.lin_out = torch.nn.Linear(inner_chan, out_chan)

    def forward(self, x):
        x = self.lin_in(x)
        x = self.gelu(x)
        x = self.lin_out(x)
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, dims, ff_width, heads):
        super().__init__()

        self.ff_norm = torch.nn.LayerNorm(dims)
        self.ff = PWFF(dims, ff_width)
        self.multihead_norm = torch.nn.LayerNorm(dims)
        self.multihead = torch.nn.MultiheadAttention(dims, heads, batch_first=True)
    
    def forward(self, in_val):
        premh = in_val
        x, _ = self.multihead(in_val, in_val, in_val)
        print(x.shape)
        x = x + premh
        x = self.multihead_norm(x)

        preff = x.clone()
        x = self.ff(x)
        x = x + preff
        x = self.ff_norm(x)

        return x

class TransformerDecoder(torch.nn.Module):
    def __init__(self, dims, ff_width, heads):
        super().__init__()

        self.ff_norm = torch.nn.LayerNorm(dims)
        self.ff = PWFF(dims, ff_width)

        self.multihead_norm = torch.nn.LayerNorm(dims)
        self.multihead = torch.nn.MultiheadAttention(dims, heads, batch_first=True)

        self.masked_mh_norm = torch.nn.LayerNorm(dims)
        self.masked_multihead = torch.nn.MultiheadAttention(dims, heads, batch_first=True)

    def forward(self, latents, encoding, pad_len):
        mask = [i < pad_len for i in range(pad_len)]
        mask = torch.tensor(mask).unsqueeze(0)
        mask = mask.expand(latents.shape[0], -1)

        premh_masked = latents
        x, _ = self.masked_multihead(latents, latents, latents, key_padding_mask=mask) # implement masked attention here
        x = x + premh_masked
        x = self.masked_mh_norm(x)

        print(x.shape, encoding.shape)
        premh = x.clone()
        x, _ = self.multihead(x, encoding, encoding)
        x = x + premh
        x = self.multihead_norm(x)

        preff = x.clone()
        x = self.ff(x)
        x = x + preff
        x = self.ff_norm(x)

        return x