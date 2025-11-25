import torch
import torch.nn as nn
import torch.nn.functional as F
import math





#################
#### Encoder ####
#################


class Encoder(nn.Module):
    """ResNet-v2 encoder with skip connections to upsampling"""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks (simplified)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)    # /4
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)   # /8
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)  # /16
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):

        layers = []
        # First block with potential stride
        layers.append(self._residual_block(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    @staticmethod
    def _residual_block(in_channels, out_channels, stride=1):
        """Basic residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):

        skips = []

        x = self.conv1(x)           # (B, 64, H/2, W/2)
        skips.append(x)

        x = self.maxpool(x)         # (B, 64, H/4, W/4)
        x = self.layer1(x)          # (B, 64, H/4, W/4)
        skips.append(x)

        x = self.layer2(x)          # (B, 128, H/8, W/8)
        skips.append(x)

        x = self.layer3(x)          # (B, 256, H/16, W/16)
        skips.append(x)

        return x, skips


###########################
#### Transformer Block ####
###########################


class PatchEmbedding(nn.Module):
    """Convert CNN feature maps to patch embeddings for transformer"""
    
    def __init__(self, in_channels: int, embed_dim: int = 768, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        
        # ViT: linear projeciton on different patching
        self.proj       = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):

        x = self.proj(x)        # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2)        # (B, embed_dim, H'*W')
        x = x.transpose(1, 2)   # (B, N, embed_dim)
        return x, (H, W)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv        = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.dropout    = nn.Dropout(dropout)
        
    def forward(self, x):

        B, N, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N)
        # q: (B, H, N, head_dim)
        # k: (B, H, head_dim, N) [after transpose]
        attn = (q @ k.transpose(-2, -1)) * self.scale   # result: (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class MLP(nn.Module):
    """MLP block for transformer"""
    
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim      = embed_dim * mlp_ratio
        self.fc1        = nn.Linear(embed_dim, hidden_dim)
        self.act        = nn.GELU()
        self.fc2        = nn.Linear(hidden_dim, embed_dim)
        self.dropout    = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block: LayerNorm -> MSA -> LayerNorm -> MLP"""
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mlp    = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


#################
#### Decoder ####
#################


class DecoderBlock(nn.Module):
    """U-net based Upsampling decoder block with transposed CNN + skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip=None):
        
        x = self.upsample(x)

        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class IntraNet2(nn.Module):
    """
    TransUNet based NN for Gene Detection
    
    Architecture:
    1. ResNet-style CNN encoder
    2. Transformer encoder (n=12 layers)
    3. Tranposed CNN for upsampling with skip connections
    """
    
    def __init__(
        self,
        img_size    : int = 224,
        in_channels : int = 1,
        num_classes : int = 2,
        
        # Transformer configs
        embed_dim   : int = 768,        # dimension for linear projection of patches
        num_layers  : int = 12,         # number of transformer layers
        num_heads   : int = 12,         # number of heads for multi-head self attention
        mlp_ratio   : int = 4,          # ratio for MLP to scale dimension upward
        dropout     : float = 0.1,
        attn_dropout: float = 0.0,
        
        # Decoder configs
        decoder_channels: list = [256, 128, 64, 32],
    ):
        super().__init__()
        
        self.img_size       = img_size
        self.num_classes    = num_classes
        self.embed_dim      = embed_dim
        
        self.encoder = Encoder(in_channels=in_channels)
        
        # Last CNN layer have 256 out_channel
        self.patch_embed = PatchEmbedding(
            in_channels =256,
            embed_dim   =embed_dim,
            patch_size  =1
        )
        
        # Positional embedding
        num_patches     = (img_size // 16) ** 2
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop   = nn.Dropout(dropout)
        
        # Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Inverse Projection to CNN feature space
        self.proj_back = nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1)
        
        # Decoder
        self.decoder1 = DecoderBlock(decoder_channels[0], 256, decoder_channels[1])  # /16  -> /8, skip from layer3
        self.decoder2 = DecoderBlock(decoder_channels[1], 128, decoder_channels[2])  # /8   -> /4, skip from layer2
        self.decoder3 = DecoderBlock(decoder_channels[2], 64, decoder_channels[3])   # /4   -> /2, skip from layer1
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], kernel_size=2, stride=2),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )  # /2 -> /1
        
        # ===== SEGMENTATION HEAD =====
        self.seg_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):

        B, C, H, W = x.shape
        
        # Encoder
        x_enc, skips = self.encoder(x)              # x_enc: (B, 256, H/16, W/16)
        
        # Transformer ; per patches
        x_patch, (h, w) = self.patch_embed(x_enc)   # (B, N, embed_dim)
        
        # Add positional embedding
        x_patch = x_patch + self.pos_embed
        x_patch = self.pos_drop(x_patch)
        
        # Apply transformer blocks
        for blk in self.transformer_blocks:
            x_patch = blk(x_patch)
        x_patch = self.norm(x_patch)
        
        # Reshape back to 2D feature map
        x_trans = x_patch.transpose(1, 2).reshape(B, self.embed_dim, h, w)  # (B, embed_dim, H/16, W/16)
        
        # Project back to decoder dimension
        x_dec = self.proj_back(x_trans)  # (B, decoder_channels[0], H/16, W/16)
        
        # Decoder U-net
        x = self.decoder1(x_dec, skips[3])   # (B, 128, H/8, W/8)
        x = self.decoder2(x, skips[2])       # (B, 64, H/4, W/4)
        x = self.decoder3(x, skips[1])       # (B, 32, H/2, W/2)
        x = self.decoder4(x)                 # (B, 32, H, W)
        
        # Segmentation
        out = self.seg_head(x)               # (B, num_classes, H, W)
        
        return out


##################
#### Auxilary ####
##################


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
