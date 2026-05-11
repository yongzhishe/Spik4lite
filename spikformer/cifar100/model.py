import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

from Spik4lite import SpikingConv2d, SpikingConv1d, prune_optimizer_state

__all__ = ['spikformer']

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        
        self.fc1_conv = SpikingConv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        
        self.fc2_conv = SpikingConv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
        self.c_hidden = hidden_features
        self.c_output = out_features

    def get_pruning_indices(self, threshold=0.5, layer_name="MLP", max_prune_rate=0.2):
        if hasattr(self.fc2_conv, 'get_average_mask_probs'):
            avg_probs = self.fc2_conv.get_average_mask_probs()
            print(f"DEBUG: {layer_name} avg_probs is {avg_probs}!!!!!!!!!") # Log removed to reduce noise
        else:
            return None

        if avg_probs is None:
            print(f"DEBUG: {layer_name} avg_probs is None (Check accumulation flow)!")
            return None

        num_channels = len(avg_probs)
        num_below_thresh = (avg_probs < threshold).sum().item()
        
        max_allowed_prune = int(num_channels * max_prune_rate)
        actual_prune_count = min(num_below_thresh, max_allowed_prune)
        num_to_keep = num_channels - actual_prune_count

        if num_to_keep == num_channels:
            return None # None means no pruning needed

        sorted_indices = torch.argsort(avg_probs, descending=True)
        keep_indices = sorted_indices[:num_to_keep]
        keep_indices, _ = keep_indices.sort()
        keep_indices = keep_indices.to(dtype=torch.long) # Use Long for indices
        
        if len(keep_indices) == 0:
             keep_indices = torch.tensor([0], device=avg_probs.device, dtype=torch.long)
             
        return keep_indices

    def apply_pruning(self, keep_indices, optimizer=None):
        if keep_indices is None:
            return self.fc1_conv.out_channels, self.fc1_conv.out_channels

        total_ch = self.fc1_conv.out_channels
        kept_ch = len(keep_indices)

        self.fc1_conv.prune_out_channels(keep_indices, optimizer=optimizer)
        self._prune_batchnorm(self.fc1_bn, keep_indices, optimizer=optimizer)
        self.fc2_conv.prune_in_channels(keep_indices, optimizer=optimizer)

        self.c_hidden = kept_ch
        return total_ch, kept_ch

    def prune_parameters(self, threshold=0.5, layer_name="MLP", max_prune_rate=0.2, optimizer=None):
        indices = self.get_pruning_indices(threshold, layer_name, max_prune_rate)
        return self.apply_pruning(indices, optimizer)

    def _prune_batchnorm(self, bn_layer, keep_indices, optimizer=None):
        keep_indices_long = keep_indices.to(device=bn_layer.weight.device, dtype=torch.long)
        with torch.no_grad():
            old_w, old_b = bn_layer.weight, bn_layer.bias
            new_w = nn.Parameter(old_w.data[keep_indices_long])
            new_b = nn.Parameter(old_b.data[keep_indices_long]) if old_b is not None else None
            
            if optimizer:
                prune_optimizer_state(optimizer, old_w, new_w, keep_indices_long, 0)
                if old_b is not None:
                    prune_optimizer_state(optimizer, old_b, new_b, keep_indices_long, 0)
            
            bn_layer.weight = new_w
            if new_b is not None: bn_layer.bias = new_b
            
            bn_layer.running_mean = bn_layer.running_mean[keep_indices_long]
            bn_layer.running_var = bn_layer.running_var[keep_indices_long]
            bn_layer.num_features = len(keep_indices)

    def forward(self, x):
        T, B, C, N = x.shape
        x = self.fc1_conv(x) 
        x = self.fc1_bn(x.flatten(0,1)).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc1_lif(x)
        
        x = self.fc2_conv(x)
        x = self.fc2_bn(x.flatten(0,1)).reshape(T, B, self.c_output, N).contiguous()
        x = self.fc2_lif(x)
        return x
    
class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        
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
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        
    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1) # [TB, C, N]
        
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,N))
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, mlp_hidden_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)

        if mlp_hidden_dim is not None:
            hidden_dim = mlp_hidden_dim
        else:
            hidden_dim = int(dim * mlp_ratio)
            
        self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + (self.mlp(x))
        return x
    
class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, structure_cfg=None):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        if structure_cfg:
            c_stage1 = structure_cfg.get('proj0', embed_dims // 8)
            c_stage2 = structure_cfg.get('proj1', embed_dims // 4)
            c_stage3 = structure_cfg.get('proj2', embed_dims // 2)
            c_final  = structure_cfg.get('proj3', embed_dims)
        else:
            c_stage1 = embed_dims // 8
            c_stage2 = embed_dims // 4
            c_stage3 = embed_dims // 2
            c_final = embed_dims

        print(f"DEBUG: SPS Init channels: {c_stage1} -> {c_stage2} -> {c_stage3} -> {c_final}")
        self.proj_conv = nn.Conv2d(in_channels, c_stage1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c_stage1)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv1 = SpikingConv2d(c_stage1, c_stage2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(c_stage2)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_conv2 = SpikingConv2d(c_stage2, c_stage3, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(c_stage3)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = SpikingConv2d(c_stage3, c_final, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(c_final)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = SpikingConv2d(c_final, c_final, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(c_final)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def get_pruning_decisions(self, threshold=0.5, max_prune_rate=0.1):
        decisions = {}

        d1 = self._calc_decision(self.proj_conv2, threshold, max_prune_rate)
        if d1 is not None: decisions['proj1'] = d1

        d2 = self._calc_decision(self.proj_conv3, threshold, max_prune_rate)
        if d2 is not None: decisions['proj2'] = d2
        
        return decisions

    def _calc_decision(self, follower_conv, threshold, max_prune_rate):
        if not hasattr(follower_conv, 'get_average_mask_probs'): return None

        avg_probs = follower_conv.get_average_mask_probs()
        print(f"DEBUG: SPS avg_probs is {avg_probs}!!!!!!!!!")
        
        if avg_probs is None: 
            return None

        num_channels = len(avg_probs)
        num_below_thresh = (avg_probs < threshold).sum().item()
        actual_prune_count = min(num_below_thresh, int(num_channels * max_prune_rate))
        num_to_keep = num_channels - actual_prune_count
        
        if num_to_keep == num_channels: return None
        
        sorted_indices = torch.argsort(avg_probs, descending=True)
        keep_indices = sorted_indices[:num_to_keep]
        keep_indices, _ = keep_indices.sort()
        return keep_indices.to(dtype=torch.long)

    def apply_pruning(self, decisions, optimizer=None):
        stats = []

        if 'proj1' in decisions:
            keep_indices = decisions['proj1']
            total_ch = self.proj_conv1.out_channels

            self.proj_conv1.prune_out_channels(keep_indices, optimizer=optimizer)
            self._prune_bn(self.proj_bn1, keep_indices, optimizer=optimizer)
            # Follower: Prune Input
            self.proj_conv2.prune_in_channels(keep_indices, optimizer=optimizer)
            
            stats.append(("proj1_out", total_ch, len(keep_indices)))

        if 'proj2' in decisions:
            keep_indices = decisions['proj2']
            total_ch = self.proj_conv2.out_channels

            self.proj_conv2.prune_out_channels(keep_indices, optimizer=optimizer)
            self._prune_bn(self.proj_bn2, keep_indices, optimizer=optimizer)
            # Follower: Prune Input
            self.proj_conv3.prune_in_channels(keep_indices, optimizer=optimizer)
            
            stats.append(("proj2_out", total_ch, len(keep_indices)))
            
        return stats

    def _prune_bn(self, bn_layer, keep_indices, optimizer=None):
        keep_indices_long = keep_indices.to(dtype=torch.long)
        with torch.no_grad():
            old_w, old_b = bn_layer.weight, bn_layer.bias
            new_w = nn.Parameter(old_w.data[keep_indices_long])
            new_b = nn.Parameter(old_b.data[keep_indices_long]) if old_b is not None else None
            
            if optimizer:
                prune_optimizer_state(optimizer, old_w, new_w, keep_indices_long, 0)
                if old_b is not None:
                    prune_optimizer_state(optimizer, old_b, new_b, keep_indices_long, 0)
                    
            bn_layer.weight = new_w
            if new_b is not None: bn_layer.bias = new_b
            bn_layer.running_mean = bn_layer.running_mean[keep_indices_long]
            bn_layer.running_var = bn_layer.running_var[keep_indices_long]
            bn_layer.num_features = len(keep_indices)

    def prune_parameters(self, threshold=0.5, max_prune_rate=0.1, optimizer=None):
        decisions = self.get_pruning_decisions(threshold, max_prune_rate)
        return self.apply_pruning(decisions, optimizer)

    def forward(self, x):
        T, B, C, H, W = x.shape
        
        x = self.proj_conv(x.flatten(0, 1)) 
        x = self.proj_bn(x).reshape(T, B, -1, x.shape[2], x.shape[3]).contiguous()
        x = self.proj_lif(x) 


        x = self.proj_conv1(x) 
        _, _, _, H_, W_ = x.shape 
        x = self.proj_bn1(x.flatten(0, 1)).reshape(T, B, -1, H_, W_).contiguous()
        x = self.proj_lif1(x)


        x = self.proj_conv2(x)
        _, _, _, H_, W_ = x.shape
        x = self.proj_bn2(x.flatten(0, 1)).reshape(T, B, -1, H_, W_).contiguous()
        x = self.proj_lif2(x)
        x = self.maxpool2(x.flatten(0,1))
        _, C_pool, H_pool, W_pool = x.shape
        x = x.view(T, B, C_pool, H_pool, W_pool)

        x = self.proj_conv3(x)
        _, _, _, H_, W_ = x.shape
        x = self.proj_bn3(x.flatten(0, 1)).reshape(T, B, -1, H_, W_).contiguous()
        x = self.proj_lif3(x)
        x = self.maxpool3(x.flatten(0,1))
        _, C_pool, H_pool, W_pool = x.shape
        x = x.view(T, B, C_pool, H_pool, W_pool)

        x_rpe = self.rpe_conv(x)
        _, _, _, H_rpe, W_rpe = x_rpe.shape
        x_rpe = self.rpe_bn(x_rpe.flatten(0,1)).reshape(T, B, -1, H_rpe, W_rpe).contiguous()
        x_rpe = self.rpe_lif(x_rpe).flatten(0,1)

        x = x.flatten(0,1)
        x = x + x_rpe
        
        C_curr = x.shape[1] 
        x = x.reshape(T, B, C_curr, -1).contiguous()
        return x
    
class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2],
                 pruned_structure_cfg=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        sps_structure_cfg = None
        blocks_structure_cfg = None

        if pruned_structure_cfg is not None:

            if 'patch_embed' in pruned_structure_cfg and 'embed' in pruned_structure_cfg['patch_embed']:
                new_embed_dim = pruned_structure_cfg['patch_embed']['embed']

                if isinstance(embed_dims, list):
                    embed_dims = new_embed_dim 
                else:
                    embed_dims = new_embed_dim

            sps_structure_cfg = pruned_structure_cfg.get('patch_embed', None)
            blocks_structure_cfg = pruned_structure_cfg.get('blocks', None)

        self.patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims,
                                 structure_cfg=sps_structure_cfg)
        
        num_patches = self.patch_embed.num_patches
        self.current_embed_dim = embed_dims

        blocks_list = []
        for j in range(depths):
            current_mlp_hidden = None
            if blocks_structure_cfg and j < len(blocks_structure_cfg):
                current_mlp_hidden = blocks_structure_cfg[j].get('mlp_hidden', None)

            blocks_list.append(Block(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratios,
                mlp_hidden_dim=current_mlp_hidden
            ))
        self.block = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dims)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _prune_bn2d(self, layer, keep_indices, optimizer):
        keep_indices = keep_indices.to(dtype=torch.long, device=layer.weight.device)
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices])
        if optimizer:
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight
      
        if layer.bias is not None:
            old_bias = layer.bias
            new_bias = nn.Parameter(old_bias.data[keep_indices])
            if optimizer:
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
            layer.bias = new_bias
          
        layer.running_mean = layer.running_mean[keep_indices]
        layer.running_var = layer.running_var[keep_indices]
        layer.num_features = len(keep_indices)

    def _prune_conv1d_input(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[:, keep_indices, :])
        if optimizer:
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)
        layer.weight = new_weight
        layer.in_channels = len(keep_indices)

    def _prune_conv1d_output(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices, :, :])
        if optimizer:
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight
        if layer.bias is not None:
            old_bias = layer.bias
            new_bias = nn.Parameter(old_bias.data[keep_indices])
            if optimizer:
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
            layer.bias = new_bias
        layer.out_channels = len(keep_indices)

    def _prune_bn1d(self, layer, keep_indices, optimizer):
        keep_indices = keep_indices.to(dtype=torch.long)
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices])
        if optimizer:
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight
        if layer.bias is not None:
            old_bias = layer.bias
            new_bias = nn.Parameter(old_bias.data[keep_indices])
            if optimizer:
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
            layer.bias = new_bias
        layer.running_mean = layer.running_mean[keep_indices]
        layer.running_var = layer.running_var[keep_indices]
        layer.num_features = len(keep_indices)

    def _prune_ln(self, layer, keep_indices, optimizer):
        keep_indices = keep_indices.to(dtype=torch.long)
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[keep_indices])
        if optimizer:
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
        layer.weight = new_weight
        old_bias = layer.bias
        new_bias = nn.Parameter(old_bias.data[keep_indices])
        if optimizer:
            prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
        layer.bias = new_bias
        if isinstance(layer.normalized_shape, int):
            layer.normalized_shape = len(keep_indices)
        else:
            layer.normalized_shape = (len(keep_indices),)

    def _prune_linear_input(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[:, keep_indices])
        if optimizer:
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)
        layer.weight = new_weight
        layer.in_features = len(keep_indices)

    def calculate_global_mask(self):
        global_scores = None
        count = 0
        for blk in self.block:
            if hasattr(blk.mlp.fc1_conv, 'get_average_mask_probs'):
                probs = blk.mlp.fc1_conv.get_average_mask_probs() 
                if probs is not None:
                    if global_scores is None: global_scores = probs
                    else: global_scores += probs
                    count += 1
        
        if global_scores is None: return None
        avg_scores = global_scores / count
        
        keep_ratio = 0.95 
        num_to_keep = max(int(len(avg_scores) * keep_ratio), 16)
        
        heads = self.block[0].attn.num_heads
        if num_to_keep < heads: 
            num_to_keep = heads
        else:
            num_to_keep = (num_to_keep // heads) * heads
        
        sorted_indices = torch.argsort(avg_scores, descending=True)
        keep_indices = sorted_indices[:num_to_keep]
        keep_indices, _ = keep_indices.sort()
        return keep_indices.to(dtype=torch.int)

    def prune_global_embedding(self, keep_indices, optimizer=None):
        print(f"Global Pruning: Keeping {len(keep_indices)} channels")
        
        # 1. Path A: RPE (SpikingConv2d)
        self.patch_embed.rpe_conv.prune_in_channels(keep_indices, optimizer) 
        self.patch_embed.rpe_conv.prune_out_channels(keep_indices, optimizer)
        self._prune_bn2d(self.patch_embed.rpe_bn, keep_indices, optimizer) 
        
        # 2. Path B: proj_conv3 (SpikingConv2d)
        self.patch_embed.proj_conv3.prune_out_channels(keep_indices, optimizer)
        self._prune_bn2d(self.patch_embed.proj_bn3, keep_indices, optimizer)

        for blk in self.block:
            self._prune_ln(blk.norm1, keep_indices, optimizer)
            self._prune_ln(blk.norm2, keep_indices, optimizer)
            
            # SSA
            self._prune_conv1d_input(blk.attn.q_conv, keep_indices, optimizer)
            self._prune_conv1d_input(blk.attn.k_conv, keep_indices, optimizer)
            self._prune_conv1d_input(blk.attn.v_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.q_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.k_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.v_conv, keep_indices, optimizer)
            self._prune_bn1d(blk.attn.q_bn, keep_indices, optimizer)
            self._prune_bn1d(blk.attn.k_bn, keep_indices, optimizer)
            self._prune_bn1d(blk.attn.v_bn, keep_indices, optimizer)
            self._prune_conv1d_input(blk.attn.proj_conv, keep_indices, optimizer)
            self._prune_conv1d_output(blk.attn.proj_conv, keep_indices, optimizer)
            self._prune_bn1d(blk.attn.proj_bn, keep_indices, optimizer)
            
            # MLP
            blk.mlp.fc1_conv.prune_in_channels(keep_indices, optimizer)
            blk.mlp.fc2_conv.prune_out_channels(keep_indices, optimizer)
            self._prune_bn1d(blk.mlp.fc2_bn, keep_indices, optimizer)

            blk.mlp.c_output = len(keep_indices)

        # 5. Final Head
        self._prune_ln(self.norm, keep_indices, optimizer)
        self._prune_linear_input(self.head, keep_indices, optimizer)
        self.current_embed_dim = len(keep_indices)

    def prune_model(self, threshold=0.5, optimizer=None, global_pruning=True,pe_max_prune_rate=0.1, mlp_max_prune_rate=0.2):
        stats = []

        global_indices = None
        if global_pruning:
            global_indices = self.calculate_global_mask()

        sps_decisions = self.patch_embed.get_pruning_decisions(threshold,max_prune_rate=pe_max_prune_rate)

        mlp_decisions_list = []
        for blk in self.block:
            indices = blk.mlp.get_pruning_indices(threshold,max_prune_rate=mlp_max_prune_rate)
            mlp_decisions_list.append(indices)

        if global_indices is not None and len(global_indices) < self.current_embed_dim:
            self.prune_global_embedding(global_indices, optimizer)
            stats.append({'layer': 'GLOBAL', 'kept': len(global_indices)})

        sps_stats = self.patch_embed.apply_pruning(sps_decisions, optimizer)
        if sps_stats: stats.extend(sps_stats)

        for i, blk in enumerate(self.block):
            indices = mlp_decisions_list[i]
            if indices is not None:
                total, kept = blk.mlp.apply_pruning(indices, optimizer)
                if total > 0:
                    stats.append({'layer': f'blk{i}_mlp', 'kept': kept, 'total': total})
                    
        return stats

    def export_structure_config(self):

        cfg = {}

        pe = self.patch_embed
        cfg['patch_embed'] = {
            'proj0': pe.proj_conv.out_channels,
            'proj1': pe.proj_conv1.out_channels,
            'proj2': pe.proj_conv2.out_channels,
            'proj3': pe.proj_conv3.out_channels, 
            'embed': self.current_embed_dim 
        }

        blocks_cfg = []
        for i, blk in enumerate(self.block):
            blocks_cfg.append({

                'mlp_hidden': blk.mlp.fc1_conv.out_channels 
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

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(4, 1, 1, 1, 1)
        x = self.patch_embed(x) 
        
        for blk in self.block:
            x = blk(x)
            
        x = x.mean(3) 
        x = x.mean(0) 
        x = self.head(x)
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(

        **kwargs
    )
    model.default_cfg = _cfg()
    return model


