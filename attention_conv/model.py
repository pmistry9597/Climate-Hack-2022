import torch
import math

class ImageEncoder(torch.nn.Module):
    def __init__(self, channel_list, kernel_size, stride=1):
        super().__init__()

        convLayers = []
        for i in range(len(channel_list)-1):
            conv = torch.nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size=kernel_size, stride=stride)
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
        #return self.block(x)
        return self.main(x)

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
#         print(x.shape)
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

    def forward(self, latents, encoding, unpadded_len):
        mask = [i >= unpadded_len for i in range(latents.shape[1])]
        mask = torch.tensor(mask).unsqueeze(0)
        mask = mask.expand(latents.shape[0], -1)

        premh_masked = latents
        x, _ = self.masked_multihead(latents, latents, latents, key_padding_mask=mask) # implement masked attention here
        x = x + premh_masked
        x = self.masked_mh_norm(x)

#         print(x.shape, encoding.shape)
        premh = x.clone()
        x, _ = self.multihead(x, encoding, encoding)
        x = x + premh
        x = self.multihead_norm(x)

        preff = x.clone()
        x = self.ff(x)
        x = x + preff
        x = self.ff_norm(x)

        return x

class Transformer(torch.nn.Module):
    def __init__(self, dims, ff_width, heads, encode_blocks, seq_len, out_seq_len):
        super().__init__()

        encoders = [TransformerEncoder(dims, ff_width, heads) for _ in range(encode_blocks)]
        self.encoders = torch.nn.Sequential(
            *encoders
        )

        decoders = [TransformerDecoder(dims, ff_width, heads) for _ in range(encode_blocks)]
        self.decoders = torch.nn.ModuleList(decoders)

        self.in_pe = self._generate_pe(seq_len, dims)
        self.out_pe = self._generate_pe(out_seq_len, dims)

        self.decodeFlatten = torch.nn.Flatten()
        self.decodeLinOut = torch.nn.Linear(out_seq_len * dims, dims)

        self.dims = dims
        self.out_seq_len = out_seq_len

    def _generate_pe(self, seq_len, dims):
        encoding = torch.empty([seq_len, dims])
        for seq in range(encoding.shape[0]):
            for dim in range(encoding.shape[1]):
                if dim % 2 == 0:
                    encoding[seq, dim] = math.sin(seq / 10000.0 ** (dim/dims))
                else:
                    encoding[seq, dim] = math.cos(seq / 10000.0 ** (dim/dims))
        
        return encoding
    
    def forward(self, in_seq):
        x = in_seq + self.in_pe
        encoding = self.encoders(x)

        out_array = torch.zeros([in_seq.shape[0], self.out_seq_len, self.dims]) #zeros represents padding

        #fill er up!!
        for entry_no in range(len(out_array)):
            for decode_pos, decoder in enumerate(self.decoders):
                decodeOut = decoder(out_array, encoding, entry_no)
                decodeOut = self.decodeFlatten(decodeOut)
                latentCode = self.decodeLinOut(decodeOut)
                out_array = out_array.transpose(0,1) # out_pos x batch_len x dims
                out_array[decode_pos] = latentCode
                out_array = out_array.transpose(0,1) # batch_len x out_pos x dims

        return out_array

class AttentionConv(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = ImageEncoder([1, 16, 32, 64, 128, 256, 512], kernel_size=3, stride=2)
        self.decoder = ImageDecoder()
        self.transformer = Transformer(512, 2048, 8, encode_blocks=6, seq_len=12, out_seq_len=24)
        
    def forward(self, in_val):
        encodings = []
        for sample in in_val:
            encoding = self.encoder(sample)
            encodings.append(encoding)
        encodings = torch.stack(encodings)
           
        x = encodings.view([-1, 12, 512])
        x = self.transformer(x)
        x = x.view([-1, 24, 512, 1, 1])
        outs = []
        for latent in x:
            out = self.decoder(latent)
            #print(out.shape)
            outs.append(out)
        outs = torch.stack(outs)
        x = outs.squeeze()
        
        return x
        
        