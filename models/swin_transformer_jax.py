from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size Tuple[int]: window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = jnp.transpose(x,  (0, 1, 3, 2, 4, 5)).reshape(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.reshape(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(B, H, W, -1)
    return x

class MLP(nn.Module):
    """Transformer MLP / feed-forward block."""
    hidden_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                    Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.normal(stddev=1e-6)
    act_layer: Optional[Type[nn.Module]] = nn.gelu

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        actual_out_dim = x.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)
        # x = nn.gelu(x)
        x = self.act_layer(x)
        x = self.dropout(x, deterministic=deterministic)
        x = nn.Dense(features=actual_out_dim, dtype=self.dtype, 
                     kernel_init=self.kernel_init, 
                     bias_init=self.bias_init)(x)
        x = self.dropout(x, deterministic=deterministic)
        return x

def log_space_continuous_position_bias(window_size):
    """
    Args:
        window_size (Tuple[int]): The height and the width of the window.

    Return:
        log_relative_position_index: (Wh*Ww, Wh*Ww, 2)
        
    """
    coords_h = jnp.arange(window_size[0])
    coords_w = jnp.arange(window_size[1])
    coords = jnp.stack(jnp.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
    coords_flatten = coords.reshape(2, -1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = jnp.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    log_relative_position_index = jnp.multiply(jnp.sign(relative_coords),
                                                jnp.log((jnp.abs(relative_coords)+1)))
    
    return log_relative_position_index

class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
        It supports both of shifted and non-shifted window.

    Attibutes:
        dim (int): Number of input channels.
        window_size (Tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop_rate (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.0
    """

    dim: int
    window_size: Tuple[int]
    num_heads: int
    qkv_bias: Optional[bool] = True
    qk_scale: Optional[float] = None
    attn_drop_rate: Optional[float] = 0.0
    proj_drop_rate: Optional[float] = 0.0
    

    def setup(self):
        self.log_relative_position_index = log_space_continuous_position_bias(self.window_size)
        self.cpb = MLP(hidden_dim=512,
                       out_dim=self.num_heads,
                       dropout_rate=0.0,
                       act_layer=nn.relu)
        # self.logit_scale = nn.Parameter(jnp.log(10 * jnp.ones((self.num_heads, 1, 1))), requires_grad=True)
        self.logit_scale = self.param('tau', nn.initializers.normal(0.02), (1,self.num_heads, 1, 1)) + jnp.log(10)
        # tau = self.param('tau', nn.initializers.normal(0.02), (1,self.num_heads, 1, 1)) + 1

        # get relative_coords_table
        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=np.float32)
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=np.float32)
        relative_coords_table = np.expand_dims(np.transpose(np.stack(
            np.meshgrid(relative_coords_h,
                            relative_coords_w)), (1, 2, 0)), axis=0)  # 1, 2*Wh-1, 2*Ww-1, 2
        
        # Haven't implemented retraining a model with a different pretrained window size
        relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        self.relative_coords_table = np.sign(relative_coords_table) * np.log2(
            np.abs(relative_coords_table) + 1.0) / np.log2(8)

        # get pair-wise relative position index for each token inside the window
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        x, y = np.meshgrid(coords_h, coords_w)
        coords = np.stack([y, x])  # 2, Wh, Ww
        coords_flatten = coords.reshape(2, -1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = np.sum(relative_coords, axis=-1)  # Wh*Ww, Wh*Wwbn 

        if self.qkv_bias:
            # keys are ignored
            self.q_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias, 
                                    #  bias_init= nn.initializers.zeros(key=random.PRNGKey(0),
                                    #                                   shape=(self.dim,)))
                                    # bias_init=nn.initializers.zeros())
                                    bias_init=nn.initializers.constant(0))
            self.v_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias, 
                                    #  bias_init= nn.initializers.zeros(key=random.PRNGKey(0),
                                    #                                   shape=(self.dim,)))
                                    # bias_init=nn.initializers.zeros())
                                    bias_init=nn.initializers.constant(0))
        else:
            self.q_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias)
            self.v_linear = nn.Dense(features=self.dim, use_bias=self.qkv_bias)
            
        # self.qkv = nn.Dense(features=self.dim*3, use_bias=qkv_bias)
        self.k_linear = nn.Dense(features=self.dim, use_bias=False)

        self.attn_drop = nn.Dropout(rate=self.attn_drop_rate)
        self.proj = nn.Dense(features=self.dim)
        self.proj_drop = nn.Dropout(rate=self.proj_drop_rate)

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic):
    # def __call__(self, x, log_relative_position_index, *, mask=None, deterministic):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        q = jnp.transpose(self.q_linear(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))[0]
        k = jnp.transpose(self.k_linear(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))[0]
        v = jnp.transpose(self.v_linear(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))[0]
        # qkv = jnp.transpose(self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads), (2, 0, 3, 1, 4))
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        qk = jnp.clip(jnp.expand_dims(jnp.linalg.norm(q, axis=-1), axis=-1)@jnp.expand_dims(jnp.linalg.norm(k, axis=-1), axis=-2), a_min=1e-6)
        attn = q@(jnp.swapaxes(k, -2,-1))/qk
        attn = attn*jnp.clip(self.logit_scale, a_min=1e-2)

        # Log-CPB
        relative_position_bias_table = self.cpb(self.relative_coords_table, deterministic=True).reshape(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.flatten()].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * nn.sigmoid(relative_position_bias)
        attn = attn + jnp.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N) + jnp.expand_dims(mask, axis=(0,2))
            attn = attn.reshape(-1, self.num_heads, N, N)
            attn = nn.softmax(attn, axis=-1)
        else:
            attn = nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn, deterministic=deterministic)

        x = jnp.swapaxes((attn @ v), 1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=deterministic)

        return x
        

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""
    
    @nn.compact
    def __call__(self, x, *, deterministic):
        return x

class AdaptiveAvgPool1d(nn.Module):
    """ 
        Applying a 1D adaptive average pooling over an input data.
    """
    output_size: int = 1

    @nn.compact
    def __call__(self, x):
        stride = (x.shape[1]//self.output_size)
        kernel_size = (x.shape[1]-(self.output_size-1)*stride)
        avg_pool = nn.avg_pool(inputs=x, window_shape=(kernel_size,), strides=(stride,))
        return avg_pool

def create_attn_mask(shift_size, input_resolution, window_size):
    if shift_size > 0:
        H, W = input_resolution
        img_mask = jnp.zeros((1, H, W, 1))
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask = img_mask.at[:, h, w, :].set(cnt)
                cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.reshape(-1, window_size[0] * window_size[1])
        attn_mask = jnp.expand_dims(mask_windows, axis=1) - jnp.expand_dims(mask_windows, axis=2)
        attn_mask = jnp.where(attn_mask==0, x=float(0.0), y=float(-100.0))
    else:
        attn_mask = None

    return attn_mask


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size Tuple[int]: Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float, optional): Dropout rate. Default: 0.0
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    dim: int
    input_resolution: Tuple[int]
    num_heads: int
    window_size: Tuple[int]
    shift_size: int = 0
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0
    act_layer: Type[nn.Module] = nn.gelu
    norm_layer: Type[nn.Module] = nn.LayerNorm

    def setup(self):
        if min(self.input_resolution) <= max(self.window_size):
            self.shift_size2 = 0
            self.window_size2 = (min(self.input_resolution), min(self.input_resolution))
        else:
            self.shift_size2 = self.shift_size
            self.window_size2 = self.window_size
        
        assert 0 <= self.shift_size2 < min(self.window_size2)
        
        self.norm1 = self.norm_layer()
        self.attn = WindowAttention(self.dim, window_size=self.window_size2,
                                     num_heads=self.num_heads, qkv_bias=self.qkv_bias,
                                     attn_drop_rate=self.attn_drop_rate, 
                                     proj_drop_rate=self.drop_rate)

        self.batch_dropout = nn.Dropout(rate=self.drop_path_rate, broadcast_dims=[1,2]) \
        if self.drop_path_rate > 0. else IdentityLayer()
        self.norm2 = self.norm_layer()
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(hidden_dim=mlp_hidden_dim, dropout_rate=self.drop_rate,
                       act_layer=self.act_layer)

        self.attn_mask = create_attn_mask(self.shift_size2, self.input_resolution, self.window_size2)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        H, W = self.input_resolution
        B, L, C = inputs.shape
        assert L == H * W
        # shortcut = x
        
        x = inputs.reshape(B, H, W, C)

        # cyclic shift
        if self.shift_size2 > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size2, -self.shift_size2), axis=(1, 2))
        else:
            shifted_x = x

        # cyclic shift
        x_windows = window_partition(shifted_x, self.window_size2)
        x_windows = x_windows.reshape(-1, self.window_size2[0] * self.window_size2[1], C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, deterministic=deterministic)

        # merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size2[0], self.window_size2[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size2, H, W)

        # reverse cyclic shift
        if self.shift_size2 > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size2, self.shift_size2), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape(B, H * W, C)

        # Swin-Transformer v2 post-norm (1)
        x = self.norm1(x)

        # Swin-Transformer v2 post-norm (1)
        x = inputs + self.batch_dropout(x, deterministic=deterministic)
        x = x + self.batch_dropout(self.norm2(self.mlp(x, deterministic=deterministic)), deterministic=deterministic)  # Swin-Transformer v2 post-norm (2)

        return x        


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    input_resolution: Tuple[int]
    dim: int
    norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm
    
    def setup(self):
        self.reduction = nn.Dense(features=2*self.dim, use_bias=False)
        self.norm = self.norm_layer()
    
    @nn.compact
    def __call__(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    dim: int
    input_resolution: Tuple[int]
    depth: int
    num_heads: int
    window_size: Tuple[int]
    mlp_ratio: float = 4
    qkv_bias: Optional[bool] = True
    drop_rate: Optional[float] = 0.0
    attn_drop_rate: Optional[float] = 0.0
    drop_path_rate: Optional[float] = 0.0
    norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm
    downsample: Type[nn.Module] = None

    def setup(self):
        self.blocks = [SwinTransformerBlock(dim=self.dim, input_resolution=self.input_resolution,
                                            num_heads=self.num_heads, window_size=self.window_size,
                                            shift_size=0 if (i % 2 == 0) else min(self.window_size) // 2,
                                            mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias,
                                            drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate,
                                            drop_path_rate=self.drop_path_rate[i] \
                                            if isinstance(self.drop_path_rate, tuple) else self.drop_path_rate,
                                            norm_layer=self.norm_layer) 
        for i in range(self.depth)]

        # patch merging layer
        if self.downsample is not None:
            self.downsample_module = self.downsample(self.input_resolution, dim=self.dim, norm_layer=self.norm_layer)
        else:
            self.downsample_module = None

    @nn.compact
    def __call__(self, x, *, deterministic):
        for blk in self.blocks:
            x = blk(x, deterministic=deterministic)
        if self.downsample is not None:
            x = self.downsample_module(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (Tuple[int]): Image size.  Default: (224, 224).
        patch_size (Tuple[int]): Patch token size. Default: (4, 4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    embed_dim: int = 96
    norm_layer: Optional[Type[nn.Module]] = None
    
    def setup(self):
        patches_resolution = [self.img_size[0] // self.patch_size[0], 
                              self.img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv(features=self.embed_dim, 
                            kernel_size=self.patch_size, 
                            strides=self.patch_size)
        
        if self.norm_layer is not None:
            self.norm_module = self.norm_layer()
        else:
            self.norm_module = None

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).reshape(B, -1, self.embed_dim) # B Ph*Pw C

        if self.norm_layer is not None:
            x = self.norm_module(x)
        return x

class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (Tuple[int]): Input image size. Default 224
        patch_size (Tuple[int]): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    img_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (4, 4)
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 96
    depths: Tuple[int] = (2, 2, 6, 2)
    num_heads: Tuple[int] = (3, 6, 12, 24)
    window_size: Tuple[int] = (7, 7)
    mlp_ratio: int = 4
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Type[nn.Module] = nn.LayerNorm
    patch_norm: bool = True

    def setup(self):
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, 
            embed_dim=self.embed_dim, norm_layer=self.norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(rate=self.drop_rate)

        self.dpr = [x.item() for x in np.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = [BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias,
                               drop_rate=self.drop_rate, attn_drop_rate=self.attn_drop_rate,
                               drop_path_rate=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=self.norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
        for i_layer in range(self.num_layers)]

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = AdaptiveAvgPool1d(1)
        self.head = nn.Dense(features=self.num_classes) if self.num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, train):
        x = self.patch_embed(x)
        x = self.pos_drop(x, deterministic=not train)

        for layer in self.layers:
            x = layer(x, deterministic=not train)

        x = self.norm(x)  # B L C
        x = self.avgpool(x)  # B C 1
        x = x.reshape(x.shape[0], -1)
        return x
    
    @nn.compact
    def __call__(self, x, *, train):
        x = self.forward_features(x, train)
        x = self.head(x)
        return x