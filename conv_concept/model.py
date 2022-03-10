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

class PWFF(torch.nn.Module):
    def __init__(self, in_chan, inner_chan, out_chan):
        super().__init__()
        self.lin_in = torch.nn.Linear(in_chan, inner_chan)
        self.gelu = torch.nn.GELU()
        self.lin_out = torch.nn.Linear(inner_chan, out_chan)

    def forward(self, x):
        x = self.lin_in(x)
        x = self.gelu(x)
        x = self.lin_out(x)
        return x

class ImageDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential( #last layer will get activated via outside since its outputting
            torch.nn.Upsample(scale_factor=8, mode='bilinear'),
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.GELU(),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(256, 128, kernel_size=3),
            torch.nn.GELU(),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(128, 64, kernel_size=3),
            torch.nn.GELU(),
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(64, 32, kernel_size=3),
            torch.nn.GELU(),     
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(32, 16, kernel_size=3),
            torch.nn.GELU(),
            torch.nn.Conv2d(16, 1, kernel_size=3),
        )

    def forward(self, x):
        return self.block(x)

class ConvConcept(torch.nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = [1, 8, 32, 64, 128, 256]
        in_kernel = 3
        in_stride = 3
        self.encoder = ImageEncoder(in_channels, in_kernel, in_stride)
        self.encodeFinal = torch.nn.Conv2d(256, 512, kernel_size=2)
        self.encodeFlatten = torch.nn.Flatten(start_dim=-2, end_dim=-1)

        self.mainFF = PWFF(6144, 6144, 1024)

        indivFFs = [PWFF(1024, 2048, 512) for _ in range(24)]
        self.indivFFs = torch.nn.ModuleList(indivFFs)

        # out_channels = [256, 128, 64, 32, 16]
        # out_kernel = 3
        # out_stride = 3
        self.decoder = ImageDecoder() #ImageDecoder(out_channels, out_kernel, out_stride)
        #self.decodeInit = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1)
        # self.decodeFinal = torch.nn.Sequential(
        #     torch.nn.Conv2d(16, 8, kernel_size=20, stride=1),
        #     torch.nn.Conv2d(8, 1, kernel_size=1, stride=1),
        # )

        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.range = 1024.0
    
    def forward(self, x):
        x = x.view([-1, 12, 128, 128])
        x = x / self.range

        encodings = []
        for sample in x:
            y = self.encoder(sample)
            y = self.encodeFinal(y)
            encodings.append(y)
        #print(encodings[0].shape)
        x = torch.stack(encodings)
        x = self.gelu(x)
        x = x.squeeze()
        x = self.encodeFlatten(x)

        x = self.mainFF(x)
        x = x.view([-1, 1024])

        #individual computations for each latent space of output image, 24 in total
        indivLatents = []
        for indivFF in self.indivFFs:
            indivLatents.append(indivFF(x))
        x = torch.stack(indivLatents)
        #dims are 12, batch_size, blah in x rn
        x = x.transpose(0, 1)

        transposeOuts = []
        for latent in x:
            y = latent.view([24, 512, 1, 1])
            #y = self.decodeInit(y)
            #y = self.gelu(y)
            y = self.decoder(y)
            #y = self.decodeFinal(y) #no activation, we're gonna use sigmoid at the end
            y = y.squeeze()
            transposeOuts.append(y)
        x = torch.stack(transposeOuts)

        x = self.sigmoid(x) * self.range

        return x