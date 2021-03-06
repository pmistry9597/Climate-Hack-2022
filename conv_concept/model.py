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
        # self.block = torch.nn.Sequential( #last layer will get activated via outside since its outputting
        #     torch.nn.Upsample(scale_factor=8, mode='bilinear'),
        #     torch.nn.Conv2d(2048, 1048, kernel_size=3),
        #     torch.nn.GELU(),
        #     torch.nn.Upsample(scale_factor=2, mode='bilinear'),
        #     torch.nn.Conv2d(1048, 512, kernel_size=3),
        #     torch.nn.GELU(),
        #     torch.nn.Upsample(scale_factor=2, mode='bilinear'),
        #     torch.nn.Conv2d(512, 256, kernel_size=3),
        #     torch.nn.GELU(),
        #     torch.nn.Upsample(scale_factor=2, mode='bilinear'),
        #     torch.nn.Conv2d(256, 64, kernel_size=3),
        #     torch.nn.GELU(),     
        #     torch.nn.Upsample(scale_factor=2, mode='bilinear'),
        #     torch.nn.Conv2d(64, 32, kernel_size=3),
        #     torch.nn.GELU(),
        #     torch.nn.Conv2d(32, 1, kernel_size=3),
        # )
        self.main = torch.nn.Sequential(
            # nz will be the input to the first convolution
            torch.nn.ConvTranspose2d(
                12*256*4, 256, kernel_size=5, 
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
        #return self.block(x)
        return self.main(x)

class ConvConcept(torch.nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = [1, 8, 32, 64, 128, 256]
        in_kernel = 3
        in_stride = 3
        self.encoder = ImageEncoder(in_channels, in_kernel, in_stride)
        # self.encodeFinal = torch.nn.Conv2d(256, 512, kernel_size=2)
        # encoders = [
        #     torch.nn.Sequential(ImageEncoder(in_channels, in_kernel, in_stride)) for _ in range(12)
        # ]
        # self.encoders = torch.nn.ModuleList(encoders)
        # self.encodeFlatten = torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self.encodeFlatten = torch.nn.Flatten(start_dim=-3, end_dim=-1)
        

        # self.mainFF = PWFF(6144, 6144, 1024)

        # indivFFs = [PWFF(1024, 2048, 512) for _ in range(24)]
        # self.indivFFs = torch.nn.ModuleList(indivFFs)

        decoders = [ImageDecoder() for _ in range(24)]
        self.decoders = torch.nn.ModuleList(decoders)
        # out_channels = [256, 128, 64, 32, 16]
        # out_kernel = 3
        # out_stride = 3
        #self.decoder = ImageDecoder() #ImageDecoder(out_channels, out_kernel, out_stride)
        #self.decodeInit = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1)
        # self.decodeFinal = torch.nn.Sequential(
        #     torch.nn.Conv2d(16, 8, kernel_size=20, stride=1),
        #     torch.nn.Conv2d(8, 1, kernel_size=1, stride=1),
        # )
        # self.denseDecode = torch.nn.Sequential(
        #     torch.nn.Linear(512, 1024),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(1024, 2048),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(2048, 64*64),
        # )

        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.range = 1024.0
    
    def forward(self, x):
        x = x.view([-1, 12, 128, 128])
        x = x / self.range

        encodings = []
        # x = x.transpose(0, 1)
        # for batch, encoder in zip(x, self.encoders):
        #     encoding = encoder(batch)
        #     encodings.append(encoding)
        for sample in x:
            y = self.encoder(sample)
            # y = self.encodeFinal(y)
            encodings.append(y)
        #print(encodings[0].shape)
        x = torch.stack(encodings)

        # x = x.transpose(0, 1)
        #print(x.shape)
        # x = self.gelu(x)
        x = x.squeeze()
        x = self.encodeFlatten(x)
        #print(x.shape)

        # x = self.mainFF(x)
        # # x = x.view([-1, 1024])
        #x = x.view([-1, 1024, 1, 1])
        x = x.view([-1, 12*256*4, 1, 1])
        print(x.shape)

        #individual computations for each latent space of output image, 24 in total
        # indivLatents = []
        # for indivFF in self.indivFFs:
        #     indivLatents.append(indivFF(x))
        # x = torch.stack(indivLatents)
        #dims are 12, batch_size, blah in x rn
        # x = x.transpose(0, 1)

        decode_outs = []
        for decoder in self.decoders:
            decode_outs.append(decoder(x))
        x = torch.stack(decode_outs)
        #print(x.shape)
        x = x.transpose(0, 1)

        # transposeOuts = []
        # for latent in x:
        #     y = latent
        #     y = y.view([24, 512, 1, 1])
        #     #y = self.decodeInit(y)
        #     #y = self.gelu(y)

        #     y = self.decoder(y)
        #     #y = self.denseDecode(y)
        #     #y = y.view([-1, 24, 64, 64])

        #     #y = self.decodeFinal(y) #no activation, we're gonna use sigmoid at the end
        #     y = y.squeeze()
        #     transposeOuts.append(y)
        #x = torch.stack(transposeOuts)
        x = x.squeeze()

        x = self.sigmoid(x) * self.range

        return x