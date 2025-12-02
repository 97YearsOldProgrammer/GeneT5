import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


#################
#### Encoder ####
#################


class Bottleneck(nn.Module):
    """
    ResNet-v2 pre-activation style CNN block
    1x1 conv -> 3x1 conv -> 1x1 conv with skip connection
    """
    
    expansion = 4  # Output channels = bottleneck_channels * expansion
    
    def __init__(
        self, 
        in_channels         : int, 
        bottleneck_channels : int, 
        stride              : int = 1,
        downsample=None
    ):
        super().__init__()
        
        out_channels = bottleneck_channels * self.expansion
        
        # Pre-activation batch norm
        self.bn1    = nn.BatchNorm1d(in_channels)
        self.conv1  = nn.Conv1d(
            in_channels, bottleneck_channels, 
            kernel_size=1, bias=False
        )
        
        self.bn2    = nn.BatchNorm1d(bottleneck_channels)
        self.conv2  = nn.Conv1d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        self.bn3    = nn.BatchNorm1d(bottleneck_channels)
        self.conv3  = nn.Conv1d(
            bottleneck_channels, out_channels,
            kernel_size=1, bias=False
        )
        
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride
    
    def forward(self, x):
        
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return out


class BasicBlock(nn.Module):
    """Basic residual block for smaller config"""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int = 1,
        downsample  : Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        
        self.bn1    = nn.BatchNorm1d(out_channels)
        self.relu   = nn.ReLU(inplace=True)
        
        self.conv2  = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2    = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.stride     = stride
    
    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """ResNetv2-style encoder following ImageNet architecture"""
    
    def __init__(
        self,
        in_channels         : int = 4, 
        layers              : List[int] = [3, 4, 23, 3],      # ResNet-101 config
        use_bottleneck      : bool = True,
        zero_init_residual  : bool = True
    ):
        super().__init__()
        
        self.use_bottleneck = use_bottleneck
        block               = Bottleneck if use_bottleneck else BasicBlock
        
        self.in_channels = 64
        
        # Conv1: Initial convolution
        self.conv1 = nn.Conv1d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1        = nn.BatchNorm1d(64)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool    = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Conv2_x: 64   -> 256
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        
        # Conv3_x: 256  -> 512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # Conv4_x: 512  -> 1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # Conv5_x: 1024 -> 2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Calculate output channels
        self.out_channels = 512 * block.expansion  # 2048 for bottleneck
        
        # Weight initialization
        self._init_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block,
        bottleneck_channels: int,
        num_blocks: int,
        stride: int = 1
    ):

        out_channels = bottleneck_channels * block.expansion
        downsample = None
        
        # Downsample if stride != 1 or channel mismatch
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        
        layers = []
        
        # First block with potential downsampling
        layers.append(block(
            self.in_channels, bottleneck_channels,
            stride=stride, downsample=downsample
        ))
        
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(
                self.in_channels, bottleneck_channels,
                stride=1, downsample=None
            ))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, zero_init_residual):

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) input tensor
        
        Returns:
            x       : (B, 2048, L/32) encoded features
            skips   : list of skip connection tensors for later upsampling
        """
        skips = []
        
        # Conv1
        x = self.conv1(x)       # (B, 64, L/2)
        x = self.bn1(x)
        x = self.relu(x)
        skips.append(x)         # Skip 0: (B, 64, L/2)
        
        x = self.maxpool(x)     # (B, 64, L/4)
        
        # Conv2_x
        x = self.layer1(x)      # (B, 256, L/4)
        skips.append(x)         # Skip 1: (B, 256, L/4)
        
        # Conv3_x
        x = self.layer2(x)      # (B, 512, L/8)
        skips.append(x)         # Skip 2: (B, 512, L/8)
        
        # Conv4_x
        x = self.layer3(x)      # (B, 1024, L/16)
        skips.append(x)         # Skip 3: (B, 1024, L/16)
        
        # Conv5_x
        x = self.layer4(x)      # (B, 2048, L/32)
        skips.append(x)         # Skip 4: (B, 2048, L/32)
        
        return x, skips


###########################
#### Transformer Block ####
###########################


class PatchEmbedding(nn.Module):
    """Convert 1D CNN feature maps to patch embeddings for transformer"""
    
    def __init__(
        self,
        in_channels : int,
        embed_dim   : int = 768,
        patch_size  : int = 1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.proj       = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) input tensor
        
        Returns:
            x: (B, N, embed_dim) patch embeddings
            L: sequence length after patching
        """
        x       = self.proj(x)        # (B, embed_dim, L')
        B, C, L = x.shape
        x       = x.transpose(1, 2)   # (B, L', embed_dim)
        return x, L


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.use_flash  = use_flash
        
        self.qkv        = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj       = nn.Linear(embed_dim, embed_dim)
        self.dropout    = nn.Dropout(dropout)
    
    def forward(self, x):
        
        B, N, D = x.shape
        
        qkv     = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv     = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Use Flash Attention when available
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        # Use scaled dot-product attention 
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out  = attn @ v
        
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class MLP(nn.Module):
    """MLP block for transformer"""
    
    def __init__(
        self,
        embed_dim   : int,
        mlp_ratio   : int = 4,
        dropout     : float = 0.0,
        activation  : str = 'gelu'
    ):
        super().__init__()
        
        hidden_dim      = embed_dim * mlp_ratio
        self.fc1        = nn.Linear(embed_dim, hidden_dim)
        self.act        = nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True)
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
    """Transformer encoder block with pre-norm architecture"""
    
    def __init__(
        self,
        embed_dim   : int,
        num_heads   : int,
        mlp_ratio   : int = 4,
        dropout     : float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mlp    = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


#################
#### Decoder ####
#################


class DecoderBlock(nn.Module):
    """
    U-net style upsampling decoder block for 1D sequences
    Transposed Conv1d + skip connections
    """
    
    def __init__(
        self,
        in_channels     : int,
        skip_channels   : int,
        out_channels    : int
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(
                out_channels + skip_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip=None):
        
        x = self.upsample(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """
    Full decoder with multiple upsampling stages
    Mirrors the encoder structure
    """
    
    def __init__(
        self,
        encoder_channels: List[int] = [64, 256, 512, 1024, 2048],
        decoder_channels: List[int] = [1024, 512, 256, 128, 64]
    ):
        super().__init__()
        
        # decoder_channels[0] receives from transformer projection
        # Skip connections from encoder (reversed order)
        
        self.decoder1 = DecoderBlock(
            decoder_channels[0], encoder_channels[4], decoder_channels[1]
        )  # 2048->1024, skip: 2048
        
        self.decoder2 = DecoderBlock(
            decoder_channels[1], encoder_channels[3], decoder_channels[2]
        )  # 1024->512, skip: 1024
        
        self.decoder3 = DecoderBlock(
            decoder_channels[2], encoder_channels[2], decoder_channels[3]
        )  # 512->256, skip: 512
        
        self.decoder4 = DecoderBlock(
            decoder_channels[3], encoder_channels[1], decoder_channels[4]
        )  # 256->128, skip: 256
        
        # Final upsampling without skip (or with conv1 skip)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose1d(
                decoder_channels[4], decoder_channels[4],
                kernel_size=2, stride=2
            ),
            nn.Conv1d(
                decoder_channels[4] + encoder_channels[0], decoder_channels[4],
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(decoder_channels[4]),
            nn.ReLU(inplace=True),
        )
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose1d(
                decoder_channels[4], decoder_channels[4],
                kernel_size=2, stride=2
            ),
            nn.Conv1d(
                decoder_channels[4], decoder_channels[4],
                kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(decoder_channels[4]),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = decoder_channels[4]
    
    def forward(self, x, skips):

        # Downsampling  : [L/2, L/4, L/8, L/16, L/32]
        # Upsampling    : [L/32, L/16, L/8, L/4, L/2]
        
        x = self.decoder1(x, skips[4])      # (B, 512, L/16)
        x = self.decoder2(x, skips[3])      # (B, 256, L/8)
        x = self.decoder3(x, skips[2])      # (B, 128, L/4)
        x = self.decoder4(x, skips[1])      # (B, 64, L/2)
        
        # Handle skip connection for decoder5
        x = F.interpolate(x, size=skips[0].shape[2], mode='linear', align_corners=False)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.decoder5[1:](x)            # (B, 64, L/2)
        
        x = self.final_upsample(x)          # (B, 64, L)
        
        return x


##########################
#### Main Model Class ####
##########################


class IntraNet(nn.Module):
    """
    TransUNet-style architecture for 1D genetic sequence analysis
    
    Architecture:
    1. ResNet-style 1D CNN encoder
    2. Transformer encoder
    3. Transposed 1D CNN decoder with skip connections
    4. Task-specific head (segmentation or MLM)
    """
    
    def __init__(
        self,
        seq_length  : int = 4096,
        in_channels : int = 1,      # kmer embedding
        num_classes : int = 2,      # For segmentation
        
        # Encoder configs
        encoder_layers: List[int] = [3, 4, 23, 3],  # ResNet-101
        use_bottleneck: bool = True,
        
        # Transformer configs
        embed_dim               : int = 768,
        num_transformer_layers  : int = 12,
        num_heads               : int = 12,
        mlp_ratio               : int = 4,
        dropout                 : float = 0.1,
        attn_dropout            : float = 0.0,
        
        # Decoder configs
        decoder_channels: List[int] = [1024, 512, 256, 128, 64],
    ):
        super().__init__()
        
        self.seq_length     = seq_length
        self.num_classes    = num_classes
        self.embed_dim      = embed_dim
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            layers=encoder_layers,
            use_bottleneck=use_bottleneck
        )
        encoder_out_channels = self.encoder.out_channels  # 2048
        
        # Encoder output channel list for skip connections
        if use_bottleneck:
            self.encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            self.encoder_channels = [64, 64, 128, 256, 512]
        
        # Patch embedding for transformer
        self.patch_embed = PatchEmbedding(
            in_channels=encoder_out_channels,
            embed_dim=embed_dim,
            patch_size=1
        )
        
        # Positional embedding
        num_patches     = seq_length // 32  # After encoder downsampling
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop   = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout
            )
            for _ in range(num_transformer_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project back to decoder dimension
        self.proj_back = nn.Conv1d(embed_dim, decoder_channels[0], kernel_size=1)
        
        # Decoder
        self.decoder = Decoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=decoder_channels
        )
        
        # Segmentation head
        self.seg_head = nn.Conv1d(
            decoder_channels[-1], num_classes,
            kernel_size=1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_transformer(self, x):
        
        B = x.shape[0]
    
        # Patch embedding
        x_patch, L = self.patch_embed(x)  # (B, L, embed_dim)
        
        # Add positional embedding
        x_patch = x_patch + self.pos_embed[:, :L, :]
        
        x_patch = self.pos_drop(x_patch)
        
        # Apply transformer blocks
        for blk in self.transformer_blocks:
            x_patch = blk(x_patch)
        
        x_patch = self.norm(x_patch)
        
        # Reshape back to 1D feature map
        x_trans = x_patch.transpose(1, 2)  # (B, embed_dim, L)
        
        return x_trans
    
    def forward_decoder(self, x, skips):
        x = self.proj_back(x)           # (B, decoder_channels[0], L)
        x = self.decoder(x, skips)
        return x
    
    def forward(self, x):

        B, C, L = x.shape
        
        # Encoder
        x_enc, skips = self.forward_encoder(x)
        
        # Transformer
        x_trans = self.forward_transformer(x_enc)
        
        # Decoder
        x_dec = self.forward_decoder(x_trans, skips)
        
        # Segmentation head
        out = self.seg_head(x_dec)
        
        # Ensure output matches input length
        if out.shape[2] != L:
            out = F.interpolate(out, size=L, mode='linear', align_corners=False)
        
        return out


##################
#### Variants ####
##################


def intranet1d_small(seq_length: int = 4096, in_channels: int = 4, num_classes: int = 2):
    """Small variant with fewer layers"""
    return IntraNet(
        seq_length=seq_length,
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_layers=[2, 2, 2, 2],  # ResNet-18 style
        use_bottleneck=False,
        embed_dim=384,
        num_transformer_layers=6,
        num_heads=6,
        decoder_channels=[256, 128, 64, 32, 16],
    )


def intranet1d_base(seq_length: int = 4096, in_channels: int = 4, num_classes: int = 2):
    """Base variant (ResNet-50 style)"""
    return IntraNet(
        seq_length=seq_length,
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_layers=[3, 4, 6, 3],  # ResNet-50
        use_bottleneck=True,
        embed_dim=768,
        num_transformer_layers=12,
        num_heads=12,
        decoder_channels=[1024, 512, 256, 128, 64],
    )


def intranet1d_large(seq_length: int = 4096, in_channels: int = 4, num_classes: int = 2):
    """Large variant (ResNet-101 style)"""
    return IntraNet(
        seq_length=seq_length,
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_layers=[3, 4, 23, 3],  # ResNet-101
        use_bottleneck=True,
        embed_dim=1024,
        num_transformer_layers=24,
        num_heads=16,
        decoder_channels=[1024, 512, 256, 128, 64],
    )


##################
#### Auxilary ####
##################


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with random input"""
    model = intranet1d_base(seq_length=4096, in_channels=4, num_classes=2)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 4, 4096)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    return model