# %%
import torch

# %%
class QKVAttention(torch.nn.Module):
    def __init__(self, scale, n, q_channels):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.scale = scale
        self.n = n
        self.q_channels = q_channels
    
    def forward(self, q, k, v): # Note: KINDA SUS, TRY TO GET RIGHT COMBO OF RESHAPE AND PERMUTE TO GET PROPER HEAD AND INDEX DIM SEPARATION
        #following permutes are sussy, intention: to get the heads together but each index dim seperate
        q = q.permute([0, 2, 1, 3])
        k = k.permute([0, 2, 1, 3])
        v = v.permute([0, 2, 1, 3])
        att_factor = torch.matmul(q, k.permute([0, 1, 3, 2]))
        att_factor = self.softmax(att_factor * self.scale)
        attention = torch.matmul(att_factor, v)
        attention = attention.permute([0, 2, 1, 3]) # intent: keep index dim together when heads are merged
        attention = attention.reshape([-1, self.n, self.q_channels])
        return attention


# %%
class MLP(torch.nn.Module):
    def __init__(self, num_channels, wide_factor):
        super().__init__()
        inner_channels = wide_factor * num_channels
        self.lin_in = torch.nn.Linear(num_channels, inner_channels) #Note: remove bias if needed
        self.lin_out = torch.nn.Linear(inner_channels, num_channels) #Note: remove bias if needed
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        x = self.lin_in(x)
        x = self.gelu(x)
        x = self.lin_out(x)
        return x

# %%
class Attention(torch.nn.Module):
    def __init__(self, num_heads, n, q_channels, kv_channels):
        super().__init__()
        if q_channels % num_heads != 0:
            raise ValueError('Bro, your damn channels don\'t work with your head count!', q_channels, num_heads)
        # if kv_channels % num_heads != 0:
        #     raise ValueError('Bro, your damn channels don\'t work with your head count!', kv_channels, num_heads)
        self.num_heads = num_heads
        self.q_head_channels = q_channels // num_heads
        
        self.q_trans = torch.nn.Linear(q_channels, q_channels)
        self.k_trans = torch.nn.Linear(kv_channels, q_channels)
        self.v_trans = torch.nn.Linear(kv_channels, q_channels)
        self.out_trans = torch.nn.Linear(q_channels, q_channels)
        self.qkvAttn = QKVAttention(q_channels ** -0.5, n, q_channels)

    def forward(self, q_in, k_in, v_in):
        q = self.q_trans(q_in)
        k = self.k_trans(k_in)
        v = self.v_trans(v_in)

        q = q.reshape([-1, q.shape[1], self.num_heads, self.q_head_channels])
        k = k.reshape([-1, k.shape[1], self.num_heads, self.q_head_channels])
        v = v.reshape([-1, v.shape[1], self.num_heads, self.q_head_channels])

        attn = self.qkvAttn(q, k, v)

        out = self.out_trans(attn)
        return out

# %%
class SelfAttention(torch.nn.Module):
    def __init__(self, n, num_channels, heads, wide_factor, p_dropout=0.1):
        super().__init__()
        self.in_norm = torch.nn.LayerNorm(num_channels)
        self.mlp_norm = torch.nn.LayerNorm(num_channels)
        self.attn = Attention(heads, n, num_channels, num_channels)
        self.mlp = MLP(num_channels, wide_factor)
        self.dropout = torch.nn.Dropout(p=p_dropout)

    def forward(self, in_val):
        x = in_val
        attn = self.in_norm(in_val)
        attn = self.attn(attn, attn, attn)
        x = x + attn

        mlp = self.mlp_norm(x)
        mlp = self.mlp(mlp)
        x = x + mlp

        return x


# %%
class CrossAttention(torch.nn.Module):
    def __init__(self, n, q_channels, kv_channels, heads, wide_factor, p_dropout=0.1, skip_att=True):
        super().__init__()
        self.q_norm = torch.nn.LayerNorm(q_channels)
        self.kv_norm = torch.nn.LayerNorm(kv_channels)
        self.mlp_norm = torch.nn.LayerNorm(q_channels)
        self.attn = Attention(heads, n, q_channels, kv_channels)
        self.mlp = MLP(q_channels, wide_factor)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.skip_att = skip_att

    def forward(self, q_kv):
        q, kv = q_kv
        x = q
        q = self.q_norm(q)
        kv = self.kv_norm(kv)
        attn = self.attn(q, kv, kv)
        if self.skip_att:
            x = x + attn
        else:
            x = attn

        mlp = self.mlp_norm(x)
        mlp = self.mlp(mlp)
        x = x + mlp

        return x

# %%
class PerceiverEncoder(torch.nn.Module):
    def __init__(self, n, q_channels, kv_channels, heads, wide_factor, latent_count, repeat_count=1, p_dropout=0.1):
        super().__init__()
        self.repeat_count = repeat_count

        latentBlocks = [SelfAttention(n, q_channels, heads, wide_factor, p_dropout) for _ in range(latent_count)]
        self.block = torch.nn.Sequential(
            CrossAttention(n, q_channels, kv_channels, heads, wide_factor, p_dropout),
            *latentBlocks
        )
    
    def forward(self, q, kv):
        x = self.block((q, kv))
        for _ in range(self.repeat_count-1):
            x = self.block((x, kv))
        return self.block((q, kv))

# %%
class PerceiverInternal(torch.nn.Module):
    def __init__(self, in_channels, latent_dim, q_out_dim, heads, wide_factor, latent_count, repeat_count=1, p_dropout=0.1):
        super().__init__()
        self.encoder = PerceiverEncoder(latent_dim[0], latent_dim[1], in_channels, heads, wide_factor, latent_count, repeat_count, p_dropout)

        q_in = torch.zeros([1, *latent_dim])
        torch.nn.init.xavier_normal_(q_in)
        self.q_in = torch.nn.Parameter(q_in)

        q_out = torch.zeros([1, *q_out_dim])
        torch.nn.init.xavier_normal_(q_out)
        self.q_out = torch.nn.Parameter(q_out)

        self.out_cross = CrossAttention(n=q_out_dim[0], q_channels=q_out_dim[1], kv_channels=latent_dim[1], heads=heads, wide_factor=wide_factor, p_dropout=p_dropout, skip_att=False)

    def forward(self, in_val):
        x = self.encoder(self.q_in, in_val)
        x = self.out_cross((self.q_out, x))
        return x

# %%
class ImagesPreprocess(torch.nn.Module): # performs embedding and positional/temporal encoding
    def __init__(self):
        super().__init__()
        #self.conv2d = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.shape = (1, 12, 128, 128)
        self.encoding = torch.nn.Parameter(self._generate_encoding())
        self.encoding.requires_grad = False

    def _generate_encoding(self):
        useful_dims = len(self.shape) - 1
        encoding = torch.zeros([*self.shape, useful_dims])
        for i in range(self.shape[1]):
            for j in range(self.shape[2]):
                for k in range(self.shape[3]):
                    encoding[0,i,j,k][0] = i * 2.0 / float(self.shape[1]) - 1.0
                    encoding[0,i,j,k][1] = j * 2.0 / float(self.shape[2]) - 1.0
                    encoding[0,i,j,k][2] = k * 2.0 / float(self.shape[3]) - 1.0
        return encoding

    def forward(self, in_val):
        x = in_val.unsqueeze(-1)
        dims = [-1 for _ in range(len(self.encoding.shape))]
        batch_count = in_val.shape[0]
        dims[0] = batch_count
        encoding = self.encoding.expand(*dims)
        x = torch.cat([x, encoding], dim=-1)

        #maintain batch and pixel codes, pixels will be in sequential form
        x = torch.flatten(x, start_dim=1, end_dim=-2)

        return x

# %%
class PerceiverCH(torch.nn.Module):
    def __init__(self, preprocessor, latent_dim, heads, wide_factor, latent_count, repeat_count=1, p_dropout=0.1):
        super().__init__()
        self.out_dim = (24, 64, 64)
        self.range = 1024.0

        self.preprocess = preprocessor
        out_dim = (self.out_dim[0], self.out_dim[1] * self.out_dim[2])
        in_channels = 1 + len(self.out_dim) #1 for pixel data, 3 for position encoding
        self.process = PerceiverInternal(in_channels, latent_dim, out_dim, heads, wide_factor, latent_count, repeat_count, p_dropout)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, in_val):
        x = in_val
        x = x / self.range
        x = self.preprocess(in_val)
        
        x = self.process(x)

        x = x.reshape([in_val.shape[0], *self.out_dim])
        x = self.sigmoid(x) * self.range

        return x

