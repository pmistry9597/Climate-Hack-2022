import torch

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, process_channels, kernel_size=3):
        super().__init__()

        self.conv0 = torch.nn.Conv2d(in_channels, process_channels, kernel_size=kernel_size)
        self.norm0 = torch.nn.BatchNorm2d(process_channels)
        self.conv1 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size)
        self.norm1 = torch.nn.BatchNorm2d(process_channels)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.gelu(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        # print(x.shape)

        return x

class Crop(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, in_val, mimicee):
        center2 = in_val.shape[2] // 2
        offset2 = mimicee.shape[2] // 2
        slice2 = slice(center2 - offset2, center2 + offset2)
        center3 = in_val.shape[3] // 2
        offset3 = mimicee.shape[3] // 2
        slice3 = slice(center3 - offset3, center3 + offset3)

        x = in_val[:, :, slice2, slice3]

        return x

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = DoubleConv(12, 256)
        self.downConv0 = DoubleConv(256, 1024)
        self.downConv1 = DoubleConv(1024, 4096)
        self.downConv2 = DoubleConv(4096, 16384)
        
        # upsample portion of network
        self.upSample0 = torch.nn.ConvTranspose2d(16384, 4096, kernel_size=2, stride=2)
        self.upConv0 = DoubleConv(4096*2, 4096)

        self.upSample1 = torch.nn.ConvTranspose2d(4096, 1024, kernel_size=2, stride=2)
        self.upConv1 = DoubleConv(1024*2, 1024)

        self.upSample2 = torch.nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
        self.upConv2 = DoubleConv(256*2, 256)

        self.upSample3 = torch.nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.upConv3 = DoubleConv(64, 64)

        self.finalConv = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 24, kernel_size=3), #NO FINAL ACTIVATION!
        )

        self.crop = Crop()

        self.sigmoid = torch.nn.Sigmoid()

        self.range = 1024.0

    def forward(self, in_val):
        x = in_val / self.range

        x0 = self.inConv(x)
        x1 = self.downConv0(self.maxpool(x0))
        x2 = self.downConv1(self.maxpool(x1))
        x3 = self.downConv2(self.maxpool(x2))
        # x1 = self.maxpool(x0)
        # x2 = self.maxpool(x1)
        # x3 = self.maxpool(x2)
#         print(x0.shape)
#         print(x1.shape)
#         print(x2.shape)
#         print(x3.shape)

        x = self.upSample0(x3)
        # print(x.shape)
        x2 = self.crop(x2, x)
        x = self.upConv0( torch.cat([x, x2], dim=1) )
    
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
        x = self.upConv1( torch.cat([x, x1], dim=1) )

        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        x = self.upConv2( torch.cat([x, x0], dim=1) )

        x = self.upSample3(x)  
        x = self.upConv3(x)
        # print(x.shape)   

        x = self.finalConv(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x