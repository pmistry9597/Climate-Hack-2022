import torch
import math

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
    
class QuadConv(torch.nn.Module):
    def __init__(self, in_channels, process_channels, kernel_size=3, padding=0):
        super().__init__()

        self.conv0 = torch.nn.Conv2d(in_channels, process_channels, kernel_size=kernel_size, padding=padding)
        self.norm0 = torch.nn.BatchNorm2d(process_channels)
        self.conv1 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = torch.nn.BatchNorm2d(process_channels)
        self.conv2 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = torch.nn.BatchNorm2d(process_channels)
        self.conv3 = torch.nn.Conv2d(process_channels, process_channels, kernel_size=kernel_size, padding=padding)
        self.norm3 = torch.nn.BatchNorm2d(process_channels)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.gelu(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu(x)
        
        x = self.conv3(x)
        
        if not (x.shape[-1] == 1 and x.shape[-2] == 1):
            x = self.norm3(x)
        
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
        
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        
#         print('after downsample')
        
        x = self.upSample0(x3)
        # print(x.shape)
        x2 = self.crop(x2, x)
        x = self.upConv0( torch.cat([x, x2], dim=1) )
        # print(x.shape, '\n')
        
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
        x = self.upConv1( torch.cat([x, x1], dim=1) )
        # print(x.shape, '\n')
        
        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        x = self.upConv2( torch.cat([x, x0], dim=1) )
        # print(x.shape, '\n')

        x = self.finalLayer(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x
    
class UNetQuad(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = QuadConv(12, 128)
        self.downConv0 = QuadConv(128, 256)
        self.downConv1 = QuadConv(256, 512)
        self.downConv2 = QuadConv(512, 8192)
        
        # upsample portion of network
        self.upSample0 = torch.nn.ConvTranspose2d(8192, 512, kernel_size=2, stride=2)
        self.upConv0 = QuadConv(8192, 512, padding=2)

        self.upSample1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upConv1 = QuadConv(512, 256)

        self.upSample2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upConv2 = QuadConv(256, 128, kernel_size=5)

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
#         print(x2.shape)
        x3 = self.downConv2(self.maxpool(x2))
        
        print(x0.shape)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        
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
    
class UNetDeep(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = DoubleConv(12, 128)
        self.downConv0 = DoubleConv(128, 256)
        self.downConv1 = DoubleConv(256, 512)
        self.downConv2 = DoubleConv(512, 1024)
        
        self.encoderFinalConv = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 2048, kernel_size=3),
            torch.nn.BatchNorm2d(2048),
            torch.nn.GELU(),
            torch.nn.Conv2d(2048, 4096, kernel_size=3),
            torch.nn.BatchNorm2d(4096),
            torch.nn.GELU(),
#             torch.nn.Conv2d(4096, 8192, kernel_size=3),
#             torch.nn.BatchNorm2d(8192),
#             torch.nn.GELU(),
        )
        
        self.decoderInitialConv = torch.nn.Sequential(
#             torch.nn.ConvTranspose2d(8192, 4096, kernel_size=3),
#             torch.nn.BatchNorm2d(4096),
#             torch.nn.GELU(),
            torch.nn.ConvTranspose2d(4096, 2048, kernel_size=3),
            torch.nn.BatchNorm2d(2048),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=3),
            torch.nn.BatchNorm2d(1024),
            torch.nn.GELU(),
        )
        
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
        
        x3 = self.encoderFinalConv(x3)
#         print(x3.shape)
        x3 = self.decoderInitialConv(x3)
        
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
        self.inConv = DoubleConv(12, 256)
        self.downConv0 = DoubleConv(256, 512)
        self.downConv1 = DoubleConv(512, 1024)
        self.downConv2 = DoubleConv(1024, 2048)
        
        # upsample portion of network
        self.upSample0 = torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upConv0 = DoubleConv(2048, 1024, padding=2)

        self.upSample1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upConv1 = DoubleConv(1024, 512)

        self.upSample2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upConv2 = DoubleConv(512, 256, kernel_size=5)

        self.finalLayer = torch.nn.Sequential(
            torch.nn.Conv2d(256, 24, kernel_size=1)
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

class AttentionUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = DoubleConv(12, 128)
        self.downConv0 = DoubleConv(128, 256)
        self.downConv1 = DoubleConv(256, 512)
        self.downConv2 = DoubleConv(512, 1024)

        # attention mechanisms for skip connections
        self.mh0 = torch.nn.MultiheadAttention(batch_first=True, num_heads=8, embed_dim=72*72, kdim=8*8, vdim=8*8)
        self.mh1 = torch.nn.MultiheadAttention(batch_first=True, num_heads=8, embed_dim=40*40, kdim=8*8, vdim=8*8)
        self.mh2 = torch.nn.MultiheadAttention(batch_first=True, num_heads=8, embed_dim=16*16, kdim=8*8, vdim=8*8)
        self.mh3 = torch.nn.MultiheadAttention(batch_first=True, num_heads=8, embed_dim=8*8, kdim=8*8, vdim=8*8)
        # layer norm for attention + res connection
        self.ln0 = torch.nn.LayerNorm(72*72)
        self.ln1 = torch.nn.LayerNorm(40*40)
        self.ln2 = torch.nn.LayerNorm(16*16)
        self.ln3 = torch.nn.LayerNorm(8*8)
        
        self.dropout = torch.nn.Dropout(0.1)
        
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
        
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        
#         print('after downsample')

        # kv tensor for all attention mechanisms
        kv = x3.view([-1, 1024, 64])
        # self attention
        q3 = kv
        x, _ = self.mh3(q3, kv, kv)
        x = self.ln3( self.dropout(x) + q3 )
        x = x.view([-1, 1024, 8, 8])
        
        x = self.upSample0(x)
        # print(x.shape)
        x2 = self.crop(x2, x)
        # attention mechanism
        q2 = x2.reshape([-1, 512, 256])
        x2, _ = self.mh2(q2, kv, kv)
        x2 = self.ln2( self.dropout(x2) + q2 )
        x2 = x2.view([-1, 512, 16, 16])
        # end of attention mechanism
        x = self.upConv0( torch.cat([x, x2], dim=1) )
#         print(x.shape)
        
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
#         x2 = self.crop(x2, x)
        # attention mechanism
        q1 = x1.reshape([-1, 256, 1600])
        x1, _ = self.mh1(q1, kv, kv)
        x1 = self.ln1( self.dropout(x1) + q1)
        x1 = x1.view([-1, 256, 40, 40])
        # end of attention mechanism
        x = self.upConv1( torch.cat([x, x1], dim=1) )
#         print(x.shape)
        
        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        # attention mechanism
        q0 = x0.reshape([-1, 128, 5184])
        x0, _ = self.mh0(q0, kv, kv)
        x0 = self.ln0( self.dropout(x0) + q0)
        x0 = x0.view([-1, 128, 72, 72])
        # end of attention mechanism
        x = self.upConv2( torch.cat([x, x0], dim=1) )
#         print(x.shape)

        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        x = self.finalLayer(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x

class Transformer(torch.nn.Module):
    def __init__(self, dims, ff_width, heads, encode_blocks, seq_len, out_seq_len):
        super().__init__()

        encoders = [TransformerEncoder(dims, ff_width, heads) for _ in range(encode_blocks)]
        self.encoders = torch.nn.Sequential(
            *encoders
        )

        self.in_pe = self._generate_pe(seq_len, dims)
        self.in_pe = torch.nn.Parameter(self.in_pe)
        self.in_pe.requires_grad = False

        self.dims = dims
        self.out_seq_len = out_seq_len

    def _generate_pe(self, seq_len, dims):
        encoding = torch.empty([seq_len, dims])
        for seq in range(encoding.shape[0]):
            for dim in range(encoding.shape[1]):
                if dim % 2 == 0:
                    encoding[seq, dim] = math.sin(seq / 10000.0 ** (dim/float(dims)))
                else:
                    encoding[seq, dim] = math.cos(seq / 10000.0 ** ((dim-1)/float(dims)))
        encoding = encoding.unsqueeze(0)
        
        return encoding
    
    def forward(self, in_seq):
        #CREATE THE EMBEDDING SPACE FOR BOTH INPUT AND OUTPUTS
        
        in_pe = self.in_pe.expand(in_seq.shape[0], -1, -1)
#         x = self.tanh(in_seq) + in_pe
        x = in_seq + in_pe
#         print(x)
        encoding = self.encoders(x)
        
        return encoding

class PWFF(torch.nn.Module):
    def __init__(self, in_chan, inner_chan, out_chan=None):
        super().__init__()
        if out_chan == None:
            out_chan = in_chan
        self.lin_in = torch.nn.Linear(in_chan, inner_chan)
#         self.gelu = torch.nn.GELU()
        self.tanh = torch.nn.Tanh()
        self.lin_out = torch.nn.Linear(inner_chan, out_chan)

    def forward(self, x):
        x = self.lin_in(x)
#         x = self.gelu(x)
        x = self.tanh(x)
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
#         print(x.shape)
        x = x + premh
        x = self.multihead_norm(x)

        preff = x.clone()
        x = self.ff(x)
        x = x + preff
        x = self.ff_norm(x)

        return x

class TransformerUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = DoubleConv(12, 128)
        self.downConv0 = DoubleConv(128, 256)
        self.downConv1 = DoubleConv(256, 512)
        self.downConv2 = DoubleConv(512, 1024)

        self.latentTransform = Transformer(8*8, 8*8*4, 8, 8, 1024, 1024)
        self.latentNorm = torch.nn.LayerNorm(8*8)
        
        self.dropout = torch.nn.Dropout(0.1)
        
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
        
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        
#         print('after downsample')

        # kv tensor for all attention mechanisms
        x3 = x3.view([-1, 1024, 64])

        # transformer on latent space + res connection
        x = self.latentTransform(x3)
        x = self.latentNorm(self.dropout(x) + x3)

        x = x.view([-1, 1024, 8, 8])
        
        x = self.upSample0(x)
        # print(x.shape)
        x2 = self.crop(x2, x)
        x = self.upConv0( torch.cat([x, x2], dim=1) )
#         print(x.shape)
        
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
        x2 = self.crop(x2, x)
        x = self.upConv1( torch.cat([x, x1], dim=1) )
#         print(x.shape)
        
        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        x = self.upConv2( torch.cat([x, x0], dim=1) )
#         print(x.shape)

        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        x = self.finalLayer(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x

class TransformerUNetNoSkip(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        
        # down sample convolutions
        self.inConv = DoubleConv(12, 128)
        self.downConv0 = DoubleConv(128, 256)
        self.downConv1 = DoubleConv(256, 512)
        self.downConv2 = DoubleConv(512, 1024)
        
        embedding = torch.empty([64, 64])
        torch.nn.init.xavier_normal_(embedding)
#         print(embedding)
        self.embedding = torch.nn.Parameter(embedding)
        
        self.latentTransform = Transformer(8*8, 8*8*4, 8, 8, 1024, 1024)
#         self.latentNorm = torch.nn.LayerNorm(8*8)
        
        self.dropout = torch.nn.Dropout(0.1)
        
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
        
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        
#         print('after downsample')

        # kv tensor for all attention mechanisms
        x3 = x3.view([-1, 1024, 64])

        # transformer on latent space
        x = torch.matmul(x3, self.embedding)
        x = self.latentTransform(x3)
#         x = self.latentNorm(self.dropout(x) + x3)

        x = x.view([-1, 1024, 8, 8])
        
        x = self.upSample0(x)
        # print(x.shape)
        x2 = self.crop(x2, x)
        x = self.upConv0( torch.cat([x, x2], dim=1) )
#         print(x.shape)
        
        x = self.upSample1(x)
        # print(x.shape)
        x1 = self.crop(x1, x)
        x2 = self.crop(x2, x)
        x = self.upConv1( torch.cat([x, x1], dim=1) )
#         print(x.shape)
        
        x = self.upSample2(x)
        # print(x.shape)
        x0 = self.crop(x0, x)
        x = self.upConv2( torch.cat([x, x0], dim=1) )
#         print(x.shape)

        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        x = self.finalLayer(x)
        # print(x.shape)

        x = self.sigmoid(x) * self.range

        return x