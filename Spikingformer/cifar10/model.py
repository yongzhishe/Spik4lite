import torch
import torch.nn as nn
from torch.nn import Module
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from functools import partial
from timm.models import create_model
import torch.nn.functional as F

from Spik4lite import SpikingConv2d, prune_optimizer_state

__all__ = ['Spikingformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.mlp1_conv = SpikingConv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.mlp2_conv = SpikingConv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def _prune_batchnorm(self, bn_layer, keep_indices, optimizer=None):
        keep_indices_device = keep_indices.to(device=bn_layer.weight.device, dtype=torch.long)
        with torch.no_grad():
            old_weight = bn_layer.weight
            new_weight = nn.Parameter(old_weight.data[keep_indices_device])
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices_device, pruning_dim=0)
            bn_layer.weight = new_weight

            if bn_layer.bias is not None:
                old_bias = bn_layer.bias
                new_bias = nn.Parameter(old_bias.data[keep_indices_device])
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices_device, pruning_dim=0)
                bn_layer.bias = new_bias

            bn_layer.num_features = len(keep_indices)
            bn_layer.running_mean = bn_layer.running_mean.data[keep_indices_device]
            bn_layer.running_var = bn_layer.running_var.data[keep_indices_device]

    def prune_parameters(self, threshold=0.5, layer_name="MLP", max_prune_rate=0.2, optimizer=None):
        if hasattr(self.mlp2_conv, 'get_average_mask_probs'):
            avg_probs = self.mlp2_conv.get_average_mask_probs()
        else:
            return 0, 0

        if avg_probs is None:
            return 0, 0

        num_channels = len(avg_probs)
        num_below_thresh = (avg_probs < threshold).sum().item()
        max_allowed_prune = int(num_channels * max_prune_rate)
        actual_prune_count = min(num_below_thresh, max_allowed_prune)
        num_to_keep = num_channels - actual_prune_count

        if num_to_keep == num_channels:
            keep_indices = torch.arange(num_channels, device=avg_probs.device, dtype=torch.int)
        else:
            sorted_indices = torch.argsort(avg_probs, descending=True)
            keep_indices = sorted_indices[:num_to_keep]
            keep_indices, _ = keep_indices.sort()
            keep_indices = keep_indices.to(dtype=torch.int)

        if len(keep_indices) == 0:
            keep_indices = torch.tensor([0], device=avg_probs.device, dtype=torch.int)

        total_ch = self.mlp1_conv.out_channels
        kept_ch = len(keep_indices)

        if kept_ch == total_ch:
            return total_ch, kept_ch

        self.mlp1_conv.prune_out_channels(keep_indices, optimizer=optimizer)
        self._prune_batchnorm(self.mlp1_bn, keep_indices, optimizer=optimizer)
        self.mlp2_conv.prune_in_channels(keep_indices, optimizer=optimizer)

        self.c_hidden = kept_ch
        return total_ch, kept_ch

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x)
        x = self.mlp1_bn(x.flatten(0, 1)).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x)
        x = self.mlp2_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        return x


class SpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)

        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class SpikingTransformer(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, forced_mlp_hidden_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        if forced_mlp_hidden_dim is not None:
            mlp_hidden_dim = forced_mlp_hidden_dim
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingTokenizer(nn.Module):

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, custom_dims=None):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        dim0 = custom_dims['block0'] if custom_dims and 'block0' in custom_dims else embed_dims // 8
        dim1 = custom_dims['block1'] if custom_dims and 'block1' in custom_dims else embed_dims // 4
        dim2 = custom_dims['block2'] if custom_dims and 'block2' in custom_dims else embed_dims // 2
        dim3 = custom_dims['block3'] if custom_dims and 'block3' in custom_dims else embed_dims // 1

        dim4 = custom_dims['block4'] if custom_dims and 'block4' in custom_dims else embed_dims

        self.block0_conv = nn.Conv2d(in_channels, dim0, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(dim0)

        self.block1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block1_conv = SpikingConv2d(dim0, dim1, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(dim1)

        self.block2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block2_conv = SpikingConv2d(dim1, dim2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(dim2)

        self.block3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block3_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = SpikingConv2d(dim2, dim3, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(dim3)

        self.block4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.block4_mp = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.block4_conv = SpikingConv2d(dim3, dim4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(dim4)

    def prune_parameters(self, threshold=0.5, max_prune_rate=0.05, optimizer=None):
        stats = []
        s1 = self._prune_layer_pair(self.block1_conv, self.block1_bn, self.block2_conv, threshold, "block1",
                                    max_prune_rate, optimizer=optimizer)
        if s1: stats.append(s1)

        s2 = self._prune_layer_pair(self.block2_conv, self.block2_bn, self.block3_conv, threshold, "block2",
                                    max_prune_rate, optimizer=optimizer)
        if s2: stats.append(s2)

        s3 = self._prune_layer_pair(self.block3_conv, self.block3_bn, self.block4_conv, threshold, "block3",
                                    max_prune_rate, optimizer=optimizer)
        if s3: stats.append(s3)
        return stats

    def _prune_layer_pair(self, leader_conv, leader_bn, follower_conv, threshold, name, max_prune_rate, optimizer=None):
        if not hasattr(follower_conv, 'get_average_mask_probs'):
            return None

        avg_probs = follower_conv.get_average_mask_probs()
        if avg_probs is None:
            return None

        num_channels = len(avg_probs)
        num_below_thresh = (avg_probs < threshold).sum().item()
        max_allowed_prune = int(num_channels * max_prune_rate)
        actual_prune_count = min(num_below_thresh, max_allowed_prune)
        num_to_keep = num_channels - actual_prune_count

        if num_to_keep == num_channels:
            keep_indices = torch.arange(num_channels, device=avg_probs.device, dtype=torch.int)
        else:
            sorted_indices = torch.argsort(avg_probs, descending=True)
            keep_indices = sorted_indices[:num_to_keep]
            keep_indices, _ = keep_indices.sort()
            keep_indices = keep_indices.to(dtype=torch.int)

        if len(keep_indices) == 0:
            keep_indices = torch.tensor([0], device=avg_probs.device, dtype=torch.int)

        total_ch = leader_conv.out_channels
        kept_ch = len(keep_indices)

        if kept_ch == total_ch:
            return None

        leader_conv.prune_out_channels(keep_indices, optimizer=optimizer)

        keep_indices_device = keep_indices.to(device=leader_bn.weight.device, dtype=torch.long)
        with torch.no_grad():
            old_weight = leader_bn.weight
            new_weight = nn.Parameter(old_weight.data[keep_indices_device])
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices_device, pruning_dim=0)
            leader_bn.weight = new_weight

            old_bias = leader_bn.bias
            new_bias = nn.Parameter(old_bias.data[keep_indices_device])
            prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices_device, pruning_dim=0)
            leader_bn.bias = new_bias

            leader_bn.running_mean = leader_bn.running_mean.data[keep_indices_device]
            leader_bn.running_var = leader_bn.running_var.data[keep_indices_device]
            leader_bn.num_features = kept_ch

        follower_conv.prune_in_channels(keep_indices, optimizer=optimizer)
        return (name, total_ch, kept_ch)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)

        x = self.block1_lif(x)
        x = self.block1_conv(x)
        x = self.block1_bn(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.block2_lif(x)
        x = self.block2_conv(x)
        x = self.block2_bn(x.flatten(0, 1)).reshape(T, B, -1, H, W)

        x = self.block3_lif(x)
        x = self.block3_mp(x.flatten(0, 1)).reshape(T, B, -1, int(H / 2), int(W / 2))
        x = self.block3_conv(x)
        x = self.block3_bn(x.flatten(0, 1)).reshape(T, B, -1, int(H / 2), int(W / 2))

        x = self.block4_lif(x)
        x = self.block4_mp(x.flatten(0, 1)).reshape(T, B, -1, int(H / 4), int(W / 4))
        x = self.block4_conv(x)
        x = self.block4_bn(x.flatten(0, 1)).reshape(T, B, -1, int(H / 4), int(W / 4))

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module):

    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4, pretrained_cfg=None,
                 pruned_structure_cfg=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        tokenizer_custom_dims = None
        if pruned_structure_cfg and 'patch_embed' in pruned_structure_cfg:
            tokenizer_custom_dims = pruned_structure_cfg['patch_embed']

        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                                       img_size_w=img_size_w,
                                       patch_size=patch_size,
                                       in_channels=in_channels,
                                       embed_dims=embed_dims,
                                       custom_dims=tokenizer_custom_dims)
        num_patches = patch_embed.num_patches

        self.embed_dims = patch_embed.block4_conv.out_channels

        blocks_list = []
        for j in range(depths):
            forced_hidden = None
            if pruned_structure_cfg and 'blocks' in pruned_structure_cfg:
                if j < len(pruned_structure_cfg['blocks']):
                    forced_hidden = pruned_structure_cfg['blocks'][j].get('mlp_hidden', None)

            blocks_list.append(SpikingTransformer(

                dim=self.embed_dims,
                num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios,
                forced_mlp_hidden_dim=forced_hidden)
            )
        block = nn.ModuleList(blocks_list)

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(self.embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _prune_conv1d_input(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[:, keep_indices, :])
        prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)
        layer.weight = new_weight
        layer.in_channels = len(keep_indices)

    def _prune_conv1d_output(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices, :, :])
        prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight

        if layer.bias is not None:
            old_bias = layer.bias
            new_bias = nn.Parameter(old_bias.data[keep_indices])
            prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
            layer.bias = new_bias
        layer.out_channels = len(keep_indices)

    def _prune_batchnorm1d(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices])
        prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight

        old_bias = layer.bias
        new_bias = nn.Parameter(old_bias.data[keep_indices])
        prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
        layer.bias = new_bias

        layer.num_features = len(keep_indices)
        layer.running_mean = layer.running_mean.data[keep_indices]
        layer.running_var = layer.running_var.data[keep_indices]

    def _prune_layernorm(self, layer, keep_indices, optimizer):
        # LayerNorm weight: [C], bias: [C]
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices])
        prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight

        old_bias = layer.bias
        new_bias = nn.Parameter(old_bias.data[keep_indices])
        prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
        layer.bias = new_bias

        # 更新 normalized_shape (对于 1D LayerNorm 这是一个 int)
        if isinstance(layer.normalized_shape, int):
            layer.normalized_shape = len(keep_indices)
        else:

            new_shape = list(layer.normalized_shape)
            new_shape[-1] = len(keep_indices)
            layer.normalized_shape = tuple(new_shape)

    def _prune_linear_input(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[:, keep_indices])
        prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)
        layer.weight = new_weight
        layer.in_features = len(keep_indices)

    def _prune_batchnorm2d(self, bn_layer, keep_indices, optimizer=None):
        keep_indices_device = keep_indices.to(device=bn_layer.weight.device, dtype=torch.long)
        with torch.no_grad():
            old_weight = bn_layer.weight
            new_weight = nn.Parameter(old_weight.data[keep_indices_device])
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices_device, pruning_dim=0)
            bn_layer.weight = new_weight

            if bn_layer.bias is not None:
                old_bias = bn_layer.bias
                new_bias = nn.Parameter(old_bias.data[keep_indices_device])
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices_device, pruning_dim=0)
                bn_layer.bias = new_bias

            bn_layer.num_features = len(keep_indices)
            bn_layer.running_mean = bn_layer.running_mean.data[keep_indices_device]
            bn_layer.running_var = bn_layer.running_var.data[keep_indices_device]

    def calculate_global_mask(self, threshold=None):
        global_scores = None
        voter_count = 0

        for blk in self.block:
            if hasattr(blk.mlp, 'mlp1_conv'):
                probs = blk.mlp.mlp1_conv.get_average_mask_probs()
                if probs is not None:
                    if global_scores is None:
                        global_scores = probs
                    else:
                        global_scores += probs
                    voter_count += 1

        if global_scores is None:
            return None

        avg_scores = global_scores / voter_count
        current_dim = len(avg_scores)

        sorted_scores, sorted_indices = torch.sort(avg_scores, descending=True)
        print(sorted_scores)

        keep_ratio = 0.5
        num_from_ratio = int(current_dim * keep_ratio)

        if threshold is None:
            scores_keep = 0
        else:
            scores_keep = (sorted_scores > threshold).sum().item()

        num_to_keep = max(num_from_ratio, 12, scores_keep)

        if hasattr(self, 'block') and len(self.block) > 0:
            first_attn = self.block[0].attn
            if hasattr(first_attn, 'num_heads'):
                heads = first_attn.num_heads

                remainder = num_to_keep % heads
                if remainder != 0:
                    num_to_keep -= remainder

                if num_to_keep < heads:
                    num_to_keep = heads

        keep_indices = sorted_indices[:num_to_keep]
        keep_indices, _ = keep_indices.sort()
        return keep_indices.to(dtype=torch.int)

    def prune_global_embedding(self, keep_indices, optimizer=None):

        self.patch_embed.block4_conv.prune_out_channels(keep_indices, optimizer=optimizer)
        self._prune_batchnorm2d(self.patch_embed.block4_bn, keep_indices, optimizer=optimizer)

        for blk in self.block:
            self._prune_layernorm(blk.norm1, keep_indices, optimizer)
            self._prune_layernorm(blk.norm2, keep_indices, optimizer)

            self._prune_conv1d_input(blk.attn.q_conv, keep_indices, optimizer)
            self._prune_conv1d_input(blk.attn.k_conv, keep_indices, optimizer)
            self._prune_conv1d_input(blk.attn.v_conv, keep_indices, optimizer)

            self._prune_conv1d_output(blk.attn.q_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.k_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.v_conv, keep_indices, optimizer)

            self._prune_batchnorm1d(blk.attn.q_bn, keep_indices, optimizer)
            self._prune_batchnorm1d(blk.attn.k_bn, keep_indices, optimizer)
            self._prune_batchnorm1d(blk.attn.v_bn, keep_indices, optimizer)

            self._prune_conv1d_input(blk.attn.proj_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.proj_conv, keep_indices, optimizer)
            self._prune_batchnorm1d(blk.attn.proj_bn, keep_indices, optimizer)

            blk.mlp.mlp1_conv.prune_in_channels(keep_indices, optimizer=optimizer)
            blk.mlp.mlp2_conv.prune_out_channels(keep_indices, optimizer=optimizer)
            self._prune_batchnorm2d(blk.mlp.mlp2_bn, keep_indices, optimizer=optimizer)

        if isinstance(self.head, nn.Linear):
            self._prune_linear_input(self.head, keep_indices, optimizer)

        self.embed_dims = len(keep_indices)

    def prune_model(self, threshold=0.5, optimizer=None, global_pruning=True,pe_max_prune_rate=0.1, mlp_max_prune_rate=0.2):

        pruning_stats = []
        GLOBAL_PRESERVE_RATIO = 0.05

        if global_pruning:
            global_indices = self.calculate_global_mask(threshold=GLOBAL_PRESERVE_RATIO)

            old_embed_dim = self.embed_dims

            if global_indices is not None and len(global_indices) < old_embed_dim:
                self.prune_global_embedding(global_indices, optimizer=optimizer)
                pruning_stats.append({'layer': 'GLOBAL_EMBEDDING',
                                      'total': old_embed_dim,
                                      'kept': self.embed_dims,
                                      'ratio': self.embed_dims / old_embed_dim})

        if hasattr(self, 'patch_embed'):
            pe_stats = self.patch_embed.prune_parameters(threshold, max_prune_rate=pe_max_prune_rate, optimizer=optimizer)
            for s in pe_stats:
                pruning_stats.append({
                    'layer': f'patch_embed.{s[0]}',
                    'total': s[1],
                    'kept': s[2],
                    'ratio': s[2] / s[1]
                })

        if hasattr(self, 'block'):
            for i, blk in enumerate(self.block):
                if hasattr(blk, 'mlp'):
                    total, kept = blk.mlp.prune_parameters(threshold, max_prune_rate=mlp_max_prune_rate, optimizer=optimizer)
                    if total > 0:
                        pruning_stats.append({
                            'layer': f'block.{i}.mlp',
                            'total': total,
                            'kept': kept,
                            'ratio': kept / total
                        })
        return pruning_stats

    def export_structure_config(self):
        cfg = {}
        pe = getattr(self, "patch_embed")
        cfg['patch_embed'] = {
            'block0': pe.block0_conv.out_channels,
            'block1': pe.block1_conv.out_channels,
            'block2': pe.block2_conv.out_channels,
            'block3': pe.block3_conv.out_channels,
            'block4': pe.block4_conv.out_channels
        }
        blocks_cfg = []
        for blk in getattr(self, "block"):
            blocks_cfg.append({
                'mlp_hidden': blk.mlp.c_hidden
            })
        cfg['blocks'] = blocks_cfg
        return cfg

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)

        for blk in block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def Spikingformer(pretrained=False, **kwargs):
    model = vit_snn(
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    input = torch.randn(2, 3, 32, 32).cuda()
    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=4, sr_ratios=1,
        T=4,
    ).cuda()

    model.eval()
    y = model(input)
    print(y.shape)
    print('Test Good!')