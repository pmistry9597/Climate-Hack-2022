import torch
from torch.nn import ConvTranspose2d

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, process_channels, kernel_size=3, padding=0):
        super().__init__()

        self.conv0 = torch.nn.Conv2d(in_channels, process_channels, kernel_size=kernel_size, padding=padding)
        self.norm0 = torch.nn.BatchNorm2d(process_channels)
        # self.conv1 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding)
        # self.norm1 = torch.nn.BatchNorm2d(process_channels)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.gelu(x)
        # print(x.shape)
        # x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.gelu(x)
        # print(x.shape)

        return x
    
class ConvBlockMulti(torch.nn.Module):
    def __init__(self, in_channels, process_channels, layers=1, kernel_size=3, paddings=0):
        super().__init__()

#         self.conv0 = 
#         self.norm0 = 
#         self.conv1 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding)
#         self.norm1 = torch.nn.BatchNorm2d(process_channels)
#         self.gelu = torch.nn.GELU()
        
        if isinstance(paddings, int):
            paddings = [ paddings for _ in range(layers) ]
            
        block = [
            torch.nn.Conv2d(in_channels, process_channels, kernel_size=kernel_size, padding=paddings[0]),
            torch.nn.BatchNorm2d(process_channels),
            torch.nn.GELU(),
        ]
        for l in range(1, layers):
            padding = paddings[l]
            block.append(torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding))
            block.append(torch.nn.BatchNorm2d(process_channels))
            block.append(torch.nn.GELU())
            
        self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        x = self.block(x)

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

class UNetPP(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2) # just reuse this lul, no learnable params here
        
        # first nodes in each layer form the encoder backbone
        self.convNodes = torch.nn.ModuleList([
            torch.nn.ModuleList([ ConvBlock(12, 128), ConvBlock(128 * 2, 128), ConvBlock(128 * 3, 128), ConvBlock(128 * 4, 128, kernel_size=5), ]), # layer 0 nodes
            torch.nn.ModuleList([ ConvBlock(128, 256), ConvBlock(256 * 2, 256), ConvBlock(256 * 3, 256, kernel_size=5), ]), # layer 1 nodes
            torch.nn.ModuleList([ ConvBlock(256, 512), ConvBlock(512 * 2, 512, kernel_size=5) ]), # layer 2 nodes
            torch.nn.ModuleList([ ConvBlock(512, 1024, kernel_size=3) ]), # layer 3 nodes
        ])
        self.nodeDiags = [ [] ]
        for d_idx in range(1, 4):
            nodeDiag = []
            for layer in self.convNodes:
                if d_idx >= len(layer):
                    continue
                nodeDiag.append( layer[d_idx] )
            self.nodeDiags.append(nodeDiag)
        
        # print(self.nodeDiags)

        self.upSamples = torch.nn.ModuleList([
            torch.nn.ModuleList([  ]), # upsample for layer 0
            torch.nn.ModuleList([ ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ]), # upsample for layer 1
            torch.nn.ModuleList([ ConvTranspose2d(512, 256, kernel_size=2, stride=2), ConvTranspose2d(512, 256, kernel_size=2, stride=2), ]), # upsample for layer 2
            torch.nn.ModuleList([ ConvTranspose2d(1024, 512, kernel_size=2, stride=2), ]), # upsample for layer 3
        ])
        self.upDiags = [ [] ]
        for d_idx in range(0, 3):
            upDiag = []
            for layer in self.upSamples:
                if d_idx >= len(layer):
                    continue
                upDiag.append( layer[d_idx] )
            self.upDiags.append(upDiag)
        
        # print(self.upDiags)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # torch.nn.Conv2d(128, 128, kernel_size=6),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.GELU(),
            # torch.nn.Conv2d(128, 128, kernel_size=6),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.GELU(),
            torch.nn.Conv2d(128, 24, kernel_size=1),
        )

        self.crop = Crop()

        self.sigmoid = torch.nn.Sigmoid()

        self.range = 1024.0

    def diagGen(self, diagsOrdered, upsOrdered, nodesOrdered, prevDiags):
        diags = []
        for d_idx in range(1, len(diagsOrdered)):            
            prevD = diagsOrdered[d_idx - 1]
            currD = diagsOrdered[d_idx]
            
            up = upsOrdered[d_idx - 1]
            node = nodesOrdered[d_idx - 1]

            upped = up(currD)
            # print('PREVD (previous elem on diagonal:', prevD.shape)
            prevD = self.crop(prevD, upped)
            
            prevDiagOuts = [ self.crop(prevDiag[d_idx - 1], upped) for prevDiag in prevDiags ]

            in_diag = torch.cat([*prevDiagOuts, prevD, upped], dim=1)
            diag = node(in_diag)

            diags.append(diag)

        # garbo = map(lambda x: x.shape, diags)
        # print(list(garbo))

        return diags

    def forward(self, in_val):
        x = in_val / self.range

        # generate initial encodings lul
        encoderLayerOuts = [] # each index corresponds to encoding for that layer
        firstEncode = True
        for layer in self.convNodes:
            if firstEncode:
                firstEncode = False
            else:
                x = self.maxpool(x)
            encoder = layer[0]
            x = encoder(x)
            # print(x.shape)
            encoderLayerOuts.append( x )
        
        # penises = map(lambda x: x.shape, encoderLayerOuts)
        # print(list(penises))

        prevDiags = []
        # first diagonal
        upDiag1 = self.upDiags[1]
        nodeDiag1 = self.nodeDiags[1]
        # print(upDiag1)
        # print(nodeDiag1)
        diag1 = self.diagGen(encoderLayerOuts, upDiag1, nodeDiag1, prevDiags)
        prevDiags.append(encoderLayerOuts)

        # second diagonal
        upDiag2 = self.upDiags[2]
        nodeDiag2 = self.nodeDiags[2]
        # print(upDiag1)
        # print(nodeDiag1)
        diag2 = self.diagGen(diag1, upDiag2, nodeDiag2, prevDiags)
        prevDiags.append(diag1)

        # penises = map(lambda x: x.shape, diag2)
        # print(list(penises))

        # third diagonal
        upDiag3 = self.upDiags[3]
        nodeDiag3 = self.nodeDiags[3]
        diag3 = self.diagGen(diag2, upDiag3, nodeDiag3, prevDiags)

        # penises = map(lambda x: x.shape, diag3)
        # print(list(penises))

        x = self.finalLayer(diag3[0])
        x = self.sigmoid(x) * self.range

        return x
    
class UNetPPDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2) # just reuse this lul, no learnable params here
        
        # first nodes in each layer form the encoder backbone
        self.convNodes = torch.nn.ModuleList([
            torch.nn.ModuleList([ ConvBlock(12, 128), ConvBlock(128 * 2, 128), ConvBlock(128 * 3, 128), ConvBlock(128 * 4, 128, kernel_size=5), ]), # layer 0 nodes
            torch.nn.ModuleList([ ConvBlock(128, 256), ConvBlock(256 * 2, 256), ConvBlock(256 * 3, 256, kernel_size=5), ]), # layer 1 nodes
            torch.nn.ModuleList([ ConvBlock(256, 512), ConvBlock(512 * 2, 512, kernel_size=5) ]), # layer 2 nodes
            torch.nn.ModuleList([ ConvBlock(512, 1024, kernel_size=3) ]), # layer 3 nodes
        ])
        self.nodeDiags = [ [] ]
        for d_idx in range(1, 4):
            nodeDiag = []
            for layer in self.convNodes:
                if d_idx >= len(layer):
                    continue
                nodeDiag.append( layer[d_idx] )
            self.nodeDiags.append(nodeDiag)
        
        # print(self.nodeDiags)

        self.upSamples = torch.nn.ModuleList([
            torch.nn.ModuleList([  ]), # upsample for layer 0
            torch.nn.ModuleList([ ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ]), # upsample for layer 1
            torch.nn.ModuleList([ ConvTranspose2d(512, 256, kernel_size=2, stride=2), ConvTranspose2d(512, 256, kernel_size=2, stride=2), ]), # upsample for layer 2
            torch.nn.ModuleList([ ConvTranspose2d(1024, 512, kernel_size=2, stride=2), ]), # upsample for layer 3
        ])
        self.upDiags = [ [] ]
        for d_idx in range(0, 3):
            upDiag = []
            for layer in self.upSamples:
                if d_idx >= len(layer):
                    continue
                upDiag.append( layer[d_idx] )
            self.upDiags.append(upDiag)
        
        # print(self.upDiags)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # torch.nn.Conv2d(128, 128, kernel_size=6),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.GELU(),
            # torch.nn.Conv2d(128, 128, kernel_size=6),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.GELU(),
            torch.nn.Conv2d(128, 24, kernel_size=1),
        )

        self.crop = Crop()

        self.sigmoid = torch.nn.Sigmoid()
        
        self.dropout = torch.nn.Dropout()
        self.range = 1024.0

    def diagGen(self, diagsOrdered, upsOrdered, nodesOrdered, prevDiags):
        diags = []
        for d_idx in range(1, len(diagsOrdered)):            
            prevD = diagsOrdered[d_idx - 1]
            currD = diagsOrdered[d_idx]
            
            up = upsOrdered[d_idx - 1]
            node = nodesOrdered[d_idx - 1]

            upped = up(currD)
            upped = self.dropout(upped)
            # print('PREVD (previous elem on diagonal:', prevD.shape)
            prevD = self.dropout(self.crop(prevD, upped))
            
            prevDiagOuts = [ self.dropout(self.crop(prevDiag[d_idx - 1], upped)) for prevDiag in prevDiags ]

            in_diag = torch.cat([*prevDiagOuts, prevD, upped], dim=1)
            diag = node(in_diag)

            diags.append(diag)

        # garbo = map(lambda x: x.shape, diags)
        # print(list(garbo))

        return diags

    def forward(self, in_val):
        x = in_val / self.range

        # generate initial encodings lul
        encoderLayerOuts = [] # each index corresponds to encoding for that layer
        firstEncode = True
        for layer in self.convNodes:
            if firstEncode:
                firstEncode = False
            else:
                x = self.maxpool(x)
                x = self.dropout(x)
            encoder = layer[0]
            x = encoder(x)
            # print(x.shape)
            encoderLayerOuts.append( x )
        
        # penises = map(lambda x: x.shape, encoderLayerOuts)
        # print(list(penises))

        prevDiags = []
        # first diagonal
        upDiag1 = self.upDiags[1]
        nodeDiag1 = self.nodeDiags[1]
        # print(upDiag1)
        # print(nodeDiag1)
        diag1 = self.diagGen(encoderLayerOuts, upDiag1, nodeDiag1, prevDiags)
        prevDiags.append(encoderLayerOuts)

        # second diagonal
        upDiag2 = self.upDiags[2]
        nodeDiag2 = self.nodeDiags[2]
        # print(upDiag1)
        # print(nodeDiag1)
        diag2 = self.diagGen(diag1, upDiag2, nodeDiag2, prevDiags)
        prevDiags.append(diag1)

        # penises = map(lambda x: x.shape, diag2)
        # print(list(penises))

        # third diagonal
        upDiag3 = self.upDiags[3]
        nodeDiag3 = self.nodeDiags[3]
        diag3 = self.diagGen(diag2, upDiag3, nodeDiag3, prevDiags)

        # penises = map(lambda x: x.shape, diag3)
        # print(list(penises))

        x = self.finalLayer(diag3[0])
        x = self.sigmoid(x) * self.range

        return x

class UNetPPDropoutMulti(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2) # just reuse this lul, no learnable params here
        
        # first nodes in each layer form the encoder backbone
        paddings = [2, 0, 0]
        self.convNodes = torch.nn.ModuleList([
            torch.nn.ModuleList([ ConvBlockMulti(12, 128, layers=3, paddings=paddings), ConvBlockMulti(128 * 2, 128, layers=3, paddings=paddings), ConvBlockMulti(128 * 3, 128, layers=3, paddings=paddings), ConvBlockMulti(128 * 4, 128, kernel_size=3, layers=3, paddings=paddings), ]), # layer 0 nodes
            torch.nn.ModuleList([ ConvBlockMulti(128, 256, layers=3, paddings=paddings), ConvBlockMulti(256 * 2, 256, layers=3, paddings=paddings), ConvBlockMulti(256 * 3, 256, kernel_size=3, layers=3, paddings=paddings), ]), # layer 1 nodes
            torch.nn.ModuleList([ ConvBlockMulti(256, 512, layers=3, paddings=paddings), ConvBlockMulti(512 * 2, 512, kernel_size=4, layers=3, paddings=paddings) ]), # layer 2 nodes
            torch.nn.ModuleList([ ConvBlockMulti(512, 1024, kernel_size=3, layers=3, paddings=paddings) ]), # layer 3 nodes
        ])
        self.nodeDiags = [ [] ]
        for d_idx in range(1, 4):
            nodeDiag = []
            for layer in self.convNodes:
                if d_idx >= len(layer):
                    continue
                nodeDiag.append( layer[d_idx] )
            self.nodeDiags.append(nodeDiag)
        
        # print(self.nodeDiags)

        self.upSamples = torch.nn.ModuleList([
            torch.nn.ModuleList([  ]), # upsample for layer 0
            torch.nn.ModuleList([ ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ]), # upsample for layer 1
            torch.nn.ModuleList([ ConvTranspose2d(512, 256, kernel_size=2, stride=2), ConvTranspose2d(512, 256, kernel_size=2, stride=2), ]), # upsample for layer 2
            torch.nn.ModuleList([ ConvTranspose2d(1024, 512, kernel_size=2, stride=2), ]), # upsample for layer 3
        ])
        self.upDiags = [ [] ]
        for d_idx in range(0, 3):
            upDiag = []
            for layer in self.upSamples:
                if d_idx >= len(layer):
                    continue
                upDiag.append( layer[d_idx] )
            self.upDiags.append(upDiag)
        
        # print(self.upDiags)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # torch.nn.Conv2d(128, 128, kernel_size=6),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.GELU(),
            torch.nn.Conv2d(128, 24, kernel_size=1),
        )

        self.crop = Crop()

        self.sigmoid = torch.nn.Sigmoid()
        
        self.dropout = torch.nn.Dropout()
        self.range = 1024.0

    def diagGen(self, diagsOrdered, upsOrdered, nodesOrdered, prevDiags):
        diags = []
        for d_idx in range(1, len(diagsOrdered)):            
            prevD = diagsOrdered[d_idx - 1]
            currD = diagsOrdered[d_idx]
            
            up = upsOrdered[d_idx - 1]
            node = nodesOrdered[d_idx - 1]

            upped = up(currD)
            upped = self.dropout(upped)
#             print('upped input shape:', upped.shape)
#             print('PREVD (previous elem on diagonal:', prevD.shape)
            prevD = self.dropout(self.crop(prevD, upped))
#             print('PREVD (previous elem on diagonal after crop:', prevD.shape)
            
            prevDiagOuts = [ self.dropout(self.crop(prevDiag[d_idx - 1], upped)) for prevDiag in prevDiags ]
#             garbo = map(lambda x: x.shape, prevDiagOuts)
#             print(list(garbo))

            in_diag = torch.cat([*prevDiagOuts, prevD, upped], dim=1)
            diag = node(in_diag)

            diags.append(diag)

        # garbo = map(lambda x: x.shape, diags)
        # print(list(garbo))

        return diags

    def forward(self, in_val):
        x = in_val / self.range

        # generate initial encodings lul
        encoderLayerOuts = [] # each index corresponds to encoding for that layer
        firstEncode = True
        for layer in self.convNodes:
            if firstEncode:
                firstEncode = False
            else:
                x = self.maxpool(x)
                x = self.dropout(x)
            encoder = layer[0]
            x = encoder(x)
            # print(x.shape)
            encoderLayerOuts.append( x )
        
        # penises = map(lambda x: x.shape, encoderLayerOuts)
        # print(list(penises))

        prevDiags = []
        # first diagonal
        upDiag1 = self.upDiags[1]
        nodeDiag1 = self.nodeDiags[1]
        # print(upDiag1)
        # print(nodeDiag1)
        diag1 = self.diagGen(encoderLayerOuts, upDiag1, nodeDiag1, prevDiags)
        prevDiags.append(encoderLayerOuts)

        # second diagonal
        upDiag2 = self.upDiags[2]
        nodeDiag2 = self.nodeDiags[2]
        # print(upDiag1)
        # print(nodeDiag1)
        diag2 = self.diagGen(diag1, upDiag2, nodeDiag2, prevDiags)
        prevDiags.append(diag1)

        # penises = map(lambda x: x.shape, diag2)
        # print(list(penises))

        # third diagonal
        upDiag3 = self.upDiags[3]
        nodeDiag3 = self.nodeDiags[3]
        diag3 = self.diagGen(diag2, upDiag3, nodeDiag3, prevDiags)

        # penises = map(lambda x: x.shape, diag3)
        # print(list(penises))

        x = self.finalLayer(diag3[0])
        x = self.sigmoid(x) * self.range

        return x
    
class UNetPPDropoutMultiBigBoi(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2) # just reuse this lul, no learnable params here
        
        # first nodes in each layer form the encoder backbone
        paddings = [2, 2, 0, 0, 0]
        layers = len(paddings)
        self.convNodes = torch.nn.ModuleList([
            torch.nn.ModuleList([ ConvBlockMulti(12, 128, layers=layers, paddings=paddings), ConvBlockMulti(128 * 2, 128, layers=layers, paddings=paddings), ConvBlockMulti(128 * 3, 128, layers=layers, paddings=paddings), ConvBlockMulti(128 * 4, 128, kernel_size=5, layers=layers, paddings=paddings), ]), # layer 0 nodes
            torch.nn.ModuleList([ ConvBlockMulti(128, 256, layers=layers, paddings=paddings), ConvBlockMulti(256 * 2, 256, layers=layers, paddings=paddings), ConvBlockMulti(256 * 3, 256, kernel_size=3, layers=layers, paddings=paddings), ]), # layer 1 nodes
            torch.nn.ModuleList([ ConvBlockMulti(256, 512, layers=layers, paddings=paddings), ConvBlockMulti(512 * 2, 512, kernel_size=3, layers=layers, paddings=paddings) ]), # layer 2 nodes
            torch.nn.ModuleList([ ConvBlockMulti(512, 1024, kernel_size=3, layers=layers, paddings=paddings) ]), # layer 3 nodes
        ])
        self.nodeDiags = [ [] ]
        for d_idx in range(1, 4):
            nodeDiag = []
            for layer in self.convNodes:
                if d_idx >= len(layer):
                    continue
                nodeDiag.append( layer[d_idx] )
            self.nodeDiags.append(nodeDiag)
        
        # print(self.nodeDiags)

        self.upSamples = torch.nn.ModuleList([
            torch.nn.ModuleList([  ]), # upsample for layer 0
            torch.nn.ModuleList([ ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ConvTranspose2d(256, 128, kernel_size=2, stride=2), ]), # upsample for layer 1
            torch.nn.ModuleList([ ConvTranspose2d(512, 256, kernel_size=2, stride=2), ConvTranspose2d(512, 256, kernel_size=2, stride=2), ]), # upsample for layer 2
            torch.nn.ModuleList([ ConvTranspose2d(1024, 512, kernel_size=2, stride=2), ]), # upsample for layer 3
        ])
        self.upDiags = [ [] ]
        for d_idx in range(0, 3):
            upDiag = []
            for layer in self.upSamples:
                if d_idx >= len(layer):
                    continue
                upDiag.append( layer[d_idx] )
            self.upDiags.append(upDiag)
        
        # print(self.upDiags)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, kernel_size=5),
            torch.nn.BatchNorm2d(128),
            torch.nn.GELU(),
            # torch.nn.Conv2d(128, 128, kernel_size=6),
            # torch.nn.BatchNorm2d(128),
            # torch.nn.GELU(),
            torch.nn.Conv2d(128, 24, kernel_size=1),
        )

        self.crop = Crop()

        self.sigmoid = torch.nn.Sigmoid()
        
        self.dropout = torch.nn.Dropout()
        self.range = 1024.0

    def diagGen(self, diagsOrdered, upsOrdered, nodesOrdered, prevDiags):
        diags = []
        for d_idx in range(1, len(diagsOrdered)):            
            prevD = diagsOrdered[d_idx - 1]
            currD = diagsOrdered[d_idx]
            
            up = upsOrdered[d_idx - 1]
            node = nodesOrdered[d_idx - 1]

            upped = up(currD)
            upped = self.dropout(upped)
#             print('upped input shape:', upped.shape)
#             print('PREVD (previous elem on diagonal:', prevD.shape)
            prevD = self.dropout(self.crop(prevD, upped))
#             print('PREVD (previous elem on diagonal after crop:', prevD.shape)
            
            prevDiagOuts = [ self.dropout(self.crop(prevDiag[d_idx - 1], upped)) for prevDiag in prevDiags ]
#             garbo = map(lambda x: x.shape, prevDiagOuts)
#             print(list(garbo))

            in_diag = torch.cat([*prevDiagOuts, prevD, upped], dim=1)
            diag = node(in_diag)

            diags.append(diag)

        # garbo = map(lambda x: x.shape, diags)
        # print(list(garbo))

        return diags

    def forward(self, in_val):
        x = in_val / self.range

        # generate initial encodings lul
        encoderLayerOuts = [] # each index corresponds to encoding for that layer
        firstEncode = True
        for layer in self.convNodes:
            if firstEncode:
                firstEncode = False
            else:
                x = self.maxpool(x)
                x = self.dropout(x)
            encoder = layer[0]
            x = encoder(x)
            # print(x.shape)
            encoderLayerOuts.append( x )
        
        # penises = map(lambda x: x.shape, encoderLayerOuts)
        # print(list(penises))

        prevDiags = []
        # first diagonal
        upDiag1 = self.upDiags[1]
        nodeDiag1 = self.nodeDiags[1]
        # print(upDiag1)
        # print(nodeDiag1)
        diag1 = self.diagGen(encoderLayerOuts, upDiag1, nodeDiag1, prevDiags)
        prevDiags.append(encoderLayerOuts)

        # second diagonal
        upDiag2 = self.upDiags[2]
        nodeDiag2 = self.nodeDiags[2]
        # print(upDiag1)
        # print(nodeDiag1)
        diag2 = self.diagGen(diag1, upDiag2, nodeDiag2, prevDiags)
        prevDiags.append(diag1)

        # penises = map(lambda x: x.shape, diag2)
        # print(list(penises))

        # third diagonal
        upDiag3 = self.upDiags[3]
        nodeDiag3 = self.nodeDiags[3]
        diag3 = self.diagGen(diag2, upDiag3, nodeDiag3, prevDiags)

        # penises = map(lambda x: x.shape, diag3)
        # print(list(penises))

        x = self.finalLayer(diag3[0])
        x = self.sigmoid(x) * self.range

        return x