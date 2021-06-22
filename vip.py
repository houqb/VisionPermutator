import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViP': _cfg(crop_pct=0.875),
    'ViP_M': _cfg(crop_pct=0.9),
    'ViP_L': _cfg(crop_pct=0.9),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WeightedSpatialMLP(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.reweight = Mlp(dim, dim // 4, dim *3)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.num_patches = num_patches # a pair


    def forward(self, x):
        B, H, W, C = x.shape

        N = C // self.num_heads
        h = x.reshape(B, H, W, self.num_heads, N).permute(0, 3, 2, 1, 4).reshape(B, self.num_heads, W, H*N)
        h = self.mlp_h(h).reshape(B, self.num_heads, W, H, N).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.num_heads, N).permute(0, 3, 1, 2, 4).reshape(B, self.num_heads, H, W*N)
        w = self.mlp_w(w).reshape(B, self.num_heads, H, W, N).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)
        
        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)  # B, C
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, shuffle_mlp_fn = WeightedSpatialMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = shuffle_mlp_fn(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = [img_size // patch_size, img_size // patch_size]
    def forward(self, x):
        x = self.proj(x) # B, C, H, W
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x) # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x

def basic_blocks(dim, num_patches, index, layers, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, \
    attn_drop=0, drop_path_rate=0., skip_lam=1.0, shuffle_mlp_fn = WeightedSpatialMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(Block(dim, num_patches, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,\
            attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, shuffle_mlp_fn = shuffle_mlp_fn))

    blocks = nn.Sequential(*blocks)

    return blocks

class ShuffleMLP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, num_heads=None, mlp_ratios=None, skip_lam=1.0,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm,shuffle_mlp_fn = WeightedSpatialMLP):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size = img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.num_patches = self.patch_embed.num_patches  # a tuple, (num_patches_h, num_patches_w)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], self.num_patches, i, layers, num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                    qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam,
                    shuffle_mlp_fn = shuffle_mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))
                self.num_patches[0] = self.num_patches[0] // patch_size
                self.num_patches[1] = self.num_patches[1] // patch_size

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self,x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))




@register_model
def vip_s14(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [False, False, False, False]
    num_heads = [16, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [384, 384, 384, 384]
    model = ShuffleMLP(layers, embed_dims=embed_dims, patch_size=14, transitions=transitions,
        num_heads=num_heads, mlp_ratios=mlp_ratios, shuffle_mlp_fn=WeightedSpatialMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP']
    return model

@register_model
def vip_s7(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [True, False, False, False]
    num_heads = [32, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [192, 384, 384, 384]
    model = ShuffleMLP(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
        num_heads=num_heads, mlp_ratios=mlp_ratios, shuffle_mlp_fn=WeightedSpatialMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP']
    return model

@register_model
def vip_m7(pretrained=False, **kwargs):
    # 55534632
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    num_heads = [32, 32, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 256, 512, 512]
    model = ShuffleMLP(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
        num_heads=num_heads, mlp_ratios=mlp_ratios, shuffle_mlp_fn=WeightedSpatialMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP_M']
    return model


@register_model
def vip_l7(pretrained=False, **kwargs):
    layers = [8, 8, 16, 4]
    transitions = [True, False, False, False]
    num_heads = [32, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 512, 512, 512]
    model = ShuffleMLP(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
        num_heads=num_heads, mlp_ratios=mlp_ratios, shuffle_mlp_fn=WeightedSpatialMLP, **kwargs)
    model.default_cfg = default_cfgs['ViP_L']
    return model
