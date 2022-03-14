import torch

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, process_channels, kernel_size=3, padding=0):
        super().__init__()

        self.conv0 = torch.nn.Conv2d(in_channels, process_channels, kernel_size=kernel_size, padding=padding)
        self.norm0 = torch.nn.BatchNorm2d(process_channels)
        self.conv1 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding)
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
        self.inConv = DoubleConv(12, 128)
        self.downConv0 = DoubleConv(128, 256)
        self.downConv1 = DoubleConv(256, 512)
        self.downConv2 = DoubleConv(512, 1024)
        
        # upsample portion of network
        self.upSample0 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upConv0 = DoubleConv(1024, 512, padding=2)

        self.upSample1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upConv1 = DoubleConv(512, 256)

        self.upSample2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upConv2 = DoubleConv(256, 128, kernel_size=5)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 24, kernel_size=1)
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
        
#         print(x0.shape)
#         print(x1.shape)
#         print(x2.shape)
#         print(x3.shape)
        
#         print('after downsample')
        
        x = self.upSample0(x3)
        # print(x.shape)
        x2 = self.crop(x2, x)
        x = self.upConv0( torch.cat([x, x2], dim=1) )
#         print(x.shape)
        
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
        x = self.upConv1( torch.cat([x, x1], dim=1) )
#         print(x.shape)
        
        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        x = self.upConv2( torch.cat([x, x0], dim=1) )
#         print(x.shape)

        x = self.finalLayer(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x



class UNetThic(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = DoubleConv(12, 512)
        self.downConv0 = DoubleConv(512, 1024)
        self.downConv1 = DoubleConv(1024, 2048)
        self.downConv2 = DoubleConv(2048, 4096)
        
        # upsample portion of network
        self.upSample0 = torch.nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=2)
        self.upConv0 = DoubleConv(4096, 2048, padding=2)

        self.upSample1 = torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upConv1 = DoubleConv(2048, 1024)

        self.upSample2 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upConv2 = DoubleConv(1024, 512, kernel_size=5)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(512, 24, kernel_size=1)
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
        
#         print(x0.shape)
#         print(x1.shape)
#         print(x2.shape)
#         print(x3.shape)
        
#         print('after downsample')
        
        x = self.upSample0(x3)
        # print(x.shape)
        x2 = self.crop(x2, x)
        x = self.upConv0( torch.cat([x, x2], dim=1) )
#         print(x.shape)
        
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
        x = self.upConv1( torch.cat([x, x1], dim=1) )
#         print(x.shape)
        
        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        x = self.upConv2( torch.cat([x, x0], dim=1) )
#         print(x.shape)

        x = self.finalLayer(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x