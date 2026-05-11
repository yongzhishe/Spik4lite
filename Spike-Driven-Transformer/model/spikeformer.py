import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from module import *

from module.Spik4lite import SpikingConv2d

def prune_optimizer_state(optimizer, old_param, new_param, keep_indices, pruning_dim):

    if optimizer is None:
        return
    if old_param not in optimizer.state:
        return

    old_state = optimizer.state[old_param]
    new_state = {}
    keep_indices = keep_indices.to(old_param.device)

    for key, value in old_state.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            if value.shape[pruning_dim] == old_param.shape[pruning_dim]:
                new_state[key] = value.index_select(pruning_dim, keep_indices).detach()
            else:
                new_state[key] = value.detach()
        else:
            new_state[key] = value

    optimizer.state[new_param] = new_state
    del optimizer.state[old_param]

    for group in optimizer.param_groups:
        for i, p in enumerate(group['params']):
            if p is old_param:
                group['params'][i] = new_param
                break


class SpikeDrivenTransformer(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=16,
            in_channels=2,
            num_classes=11,
            embed_dims=512,
            num_heads=8,
            mlp_ratios=4,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=[6, 8, 6],
            sr_ratios=[8, 4, 2],
            T=4,
            pooling_stat="1111",
            attn_mode="direct_xor",
            spike_mode="lif",
            get_embed=False,
            dvs_mode=False,
            TET=False,
            cml=False,
            pretrained=False,
            pretrained_cfg=None,

            pruned_structure_cfg=None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        tokenizer_custom_dims = None
        if pruned_structure_cfg and 'patch_embed' in pruned_structure_cfg:
            tokenizer_custom_dims = pruned_structure_cfg['patch_embed']

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
            custom_dims=tokenizer_custom_dims
        )

        if hasattr(patch_embed, 'block4_conv'):
            self.embed_dims = patch_embed.block4_conv.out_channels
        elif hasattr(patch_embed, 'rpe_conv'):
            self.embed_dims = patch_embed.rpe_conv.out_channels
        else:
            self.embed_dims = embed_dims

        blocks_list = []
        for j in range(depths):
            forced_hidden = None
            if pruned_structure_cfg and 'blocks' in pruned_structure_cfg:
                if j < len(pruned_structure_cfg['blocks']):
                    forced_hidden = pruned_structure_cfg['blocks'][j].get('mlp_hidden', None)

            blocks_list.append(
                MS_Block_Conv(
                    dim=self.embed_dims,  
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                    forced_mlp_hidden_dim=forced_hidden
                )
            )
        blocks = nn.ModuleList(blocks_list)

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.head = (
            nn.Linear(self.embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _prune_linear_input(self, layer, keep_indices, optimizer):
        old_weight = layer.weight
        new_weight = nn.Parameter(old_weight.data[:, keep_indices])
        prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)
        layer.weight = new_weight
        layer.in_features = len(keep_indices)

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

        if isinstance(layer.normalized_shape, int):
            layer.normalized_shape = len(keep_indices)
        else:
            new_shape = list(layer.normalized_shape)
            new_shape[-1] = len(keep_indices)
            layer.normalized_shape = tuple(new_shape)

    def _prune_layer_input(self, layer, keep_indices, optimizer):

        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            old_weight = layer.weight

            new_weight = nn.Parameter(old_weight.data.index_select(1, keep_indices))
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)
            layer.weight = new_weight
            layer.in_channels = len(keep_indices)
        elif isinstance(layer, nn.Linear):
            self._prune_linear_input(layer, keep_indices, optimizer)

    def _prune_layer_output(self, layer, keep_indices, optimizer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            old_weight = layer.weight

            new_weight = nn.Parameter(old_weight.data.index_select(0, keep_indices))
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
            layer.weight = new_weight

            if layer.bias is not None:
                old_bias = layer.bias
                new_bias = nn.Parameter(old_bias.data[keep_indices])
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
                layer.bias = new_bias
            layer.out_channels = len(keep_indices)
        elif isinstance(layer, nn.Linear):

            old_weight = layer.weight
            new_weight = nn.Parameter(old_weight.data.index_select(0, keep_indices))
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
            layer.weight = new_weight

            if layer.bias is not None:
                old_bias = layer.bias
                new_bias = nn.Parameter(old_bias.data.index_select(0, keep_indices))
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
                layer.bias = new_bias
            layer.out_features = len(keep_indices)

    def _prune_batchnorm(self, layer, keep_indices, optimizer):
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            old_weight = layer.weight
            new_weight = nn.Parameter(old_weight.data[keep_indices])
            prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)
            layer.weight = new_weight

            if layer.bias is not None:
                old_bias = layer.bias
                new_bias = nn.Parameter(old_bias.data[keep_indices])
                prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
                layer.bias = new_bias

            layer.num_features = len(keep_indices)
            layer.running_mean = layer.running_mean.data[keep_indices]
            layer.running_var = layer.running_var.data[keep_indices]

    def calculate_global_mask(self, threshold=None):
        global_scores = None
        voter_count = 0

        for blk in self.block:
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'mlp1_conv') and isinstance(blk.mlp.mlp1_conv, SpikingConv2d):
                probs = blk.mlp.mlp1_conv.get_average_mask_probs()
                if probs is not None:
                    if global_scores is None:
                        global_scores = probs
                    else:

                        global_scores += probs
                    voter_count += 1
        
        pe = self.patch_embed
        rpe_voter = None
        if hasattr(pe, 'rpe_conv') and isinstance(pe.rpe_conv, SpikingConv2d):
            rpe_voter = pe.rpe_conv
        elif hasattr(pe, 'block4_conv') and isinstance(pe.block4_conv, SpikingConv2d):
            rpe_voter = pe.block4_conv
            
        if rpe_voter is not None:
            probs = rpe_voter.get_average_mask_probs()

            if probs is not None:
                if global_scores is None:
                    global_scores = probs
                else:
                    if probs.shape == global_scores.shape:
                        global_scores += probs
                        voter_count += 1
                    else:
                        print(f"Warning: RPE voter shape {probs.shape} mismatch with global {global_scores.shape}")

        if global_scores is None:
            return None

        avg_scores = global_scores / voter_count
        current_dim = len(avg_scores)

        sorted_scores, sorted_indices = torch.sort(avg_scores, descending=True)

        keep_ratio = 0.98 
        num_to_keep = int(current_dim * keep_ratio)
        


        num_to_keep = max(num_to_keep, 16)

        if hasattr(self, 'block') and len(self.block) > 0:
            if hasattr(self.block[0], 'attn') and hasattr(self.block[0].attn, 'num_heads'):
                heads = self.block[0].attn.num_heads
                remainder = num_to_keep % heads
                if remainder != 0:
                    num_to_keep -= remainder
                if num_to_keep < heads:
                    num_to_keep = heads

        keep_indices = sorted_indices[:num_to_keep]
        keep_indices, _ = keep_indices.sort()

        
        return keep_indices.to(dtype=torch.int)

    def prune_global_embedding(self, keep_indices, optimizer=None):
        print(f"Executing Global Pruning: Keeping {len(keep_indices)} channels...")

        pe = self.patch_embed

        if hasattr(pe, 'block3_conv') and isinstance(pe.block3_conv, SpikingConv2d):
            pe.block3_conv.prune_out_channels(keep_indices, optimizer=optimizer)
        elif hasattr(pe, 'proj_conv3') and isinstance(pe.proj_conv3, SpikingConv2d):
            pe.proj_conv3.prune_out_channels(keep_indices, optimizer=optimizer)

        if hasattr(pe, 'proj_bn3'):
            self._prune_batchnorm(pe.proj_bn3, keep_indices, optimizer=optimizer)
        elif hasattr(pe, 'block3_bn'): # Fallback assumption
             self._prune_batchnorm(pe.block3_bn, keep_indices, optimizer=optimizer)

        if hasattr(pe, 'block4_conv') and isinstance(pe.block4_conv, SpikingConv2d):
            pe.block4_conv.prune_in_channels(keep_indices, optimizer=optimizer)
        elif hasattr(pe, 'rpe_conv') and isinstance(pe.rpe_conv, SpikingConv2d):
            pe.rpe_conv.prune_in_channels(keep_indices, optimizer=optimizer)

        if hasattr(pe, 'block4_conv'):
            pe.block4_conv.prune_out_channels(keep_indices, optimizer=optimizer)
        elif hasattr(pe, 'rpe_conv'):
            pe.rpe_conv.prune_out_channels(keep_indices, optimizer=optimizer)

        if hasattr(pe, 'block4_bn'):
            self._prune_batchnorm(pe.block4_bn, keep_indices, optimizer=optimizer)
        elif hasattr(pe, 'rpe_bn'):
            self._prune_batchnorm(pe.rpe_bn, keep_indices, optimizer=optimizer)

        for blk in self.block:
            # LayerNorms
            if hasattr(blk, 'norm1'): self._prune_layernorm(blk.norm1, keep_indices, optimizer)
            if hasattr(blk, 'norm2'): self._prune_layernorm(blk.norm2, keep_indices, optimizer)

            if hasattr(blk, 'attn'):
                attn = blk.attn
                # QKV Input
                if hasattr(attn, 'q_conv'): self._prune_layer_input(attn.q_conv, keep_indices, optimizer)
                if hasattr(attn, 'k_conv'): self._prune_layer_input(attn.k_conv, keep_indices, optimizer)
                if hasattr(attn, 'v_conv'): self._prune_layer_input(attn.v_conv, keep_indices, optimizer)
                
                # QKV Output
                if hasattr(attn, 'q_conv'): self._prune_layer_output(attn.q_conv, keep_indices, optimizer)
                if hasattr(attn, 'k_conv'): self._prune_layer_output(attn.k_conv, keep_indices, optimizer)
                if hasattr(attn, 'v_conv'): self._prune_layer_output(attn.v_conv, keep_indices, optimizer)

                # BN
                if hasattr(attn, 'q_bn'): self._prune_batchnorm(attn.q_bn, keep_indices, optimizer)
                if hasattr(attn, 'k_bn'): self._prune_batchnorm(attn.k_bn, keep_indices, optimizer)
                if hasattr(attn, 'v_bn'): self._prune_batchnorm(attn.v_bn, keep_indices, optimizer)

                # Projection Input & Output
                if hasattr(attn, 'proj_conv'):
                    self._prune_layer_input(attn.proj_conv, keep_indices, optimizer)
                    self._prune_layer_output(attn.proj_conv, keep_indices, optimizer)
                if hasattr(attn, 'proj'):  # Linear case
                    self._prune_layer_input(attn.proj, keep_indices, optimizer)
                    self._prune_layer_output(attn.proj, keep_indices, optimizer)

                if hasattr(attn, 'proj_bn'): self._prune_batchnorm(attn.proj_bn, keep_indices, optimizer)

            if hasattr(blk, 'mlp'):

                if hasattr(blk.mlp, 'mlp1_conv') and isinstance(blk.mlp.mlp1_conv, SpikingConv2d):
                    blk.mlp.mlp1_conv.prune_in_channels(keep_indices, optimizer=optimizer)
                elif hasattr(blk.mlp, 'fc1'):
                    self._prune_layer_input(blk.mlp.fc1, keep_indices, optimizer)

                if hasattr(blk.mlp, 'mlp2_conv') and isinstance(blk.mlp.mlp2_conv, SpikingConv2d):
                    blk.mlp.mlp2_conv.prune_out_channels(keep_indices, optimizer=optimizer)
                    if hasattr(blk.mlp, 'mlp2_bn'):
                        self._prune_batchnorm(blk.mlp.mlp2_bn, keep_indices, optimizer)
                elif hasattr(blk.mlp, 'fc2'):
                    self._prune_layer_output(blk.mlp.fc2, keep_indices, optimizer)

        # 3. Head
        if hasattr(self, 'head') and isinstance(self.head, nn.Linear):
            self._prune_linear_input(self.head, keep_indices, optimizer)

        self.embed_dims = len(keep_indices)

    def prune_model(self, threshold=0.5, optimizer=None, global_pruning=True,pe_max_prune_rate=0.1, mlp_max_prune_rate=0.2):

        pruning_stats = []

        if global_pruning:
            global_indices = self.calculate_global_mask(threshold=0.97)
            old_embed_dim = self.embed_dims

            if global_indices is not None and len(global_indices) < old_embed_dim:
                self.prune_global_embedding(global_indices, optimizer=optimizer)
                pruning_stats.append({'layer': 'GLOBAL_EMBEDDING',
                                      'total': old_embed_dim,
                                      'kept': self.embed_dims,
                                      'ratio': self.embed_dims / old_embed_dim})

        if hasattr(self, 'patch_embed') and hasattr(self.patch_embed, 'prune_parameters'):
            pe_stats = self.patch_embed.prune_parameters(threshold, optimizer=optimizer,max_prune_rate=pe_max_prune_rate, )
            if pe_stats:
                for s in pe_stats:
                    pruning_stats.append({
                        'layer': f'patch_embed.{s[0]}',
                        'total': s[1],
                        'kept': s[2],
                        'ratio': s[2] / s[1]
                    })

        if hasattr(self, 'block'):
            for i, blk in enumerate(self.block):
                if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'prune_parameters'):
                    total, kept = blk.mlp.prune_parameters(threshold, optimizer=optimizer,max_prune_rate=mlp_max_prune_rate)
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
        # Patch Embed
        if hasattr(self, 'patch_embed'):
            pe = self.patch_embed

            cfg['patch_embed'] = {}
            if hasattr(pe, 'block0_conv'): cfg['patch_embed']['block0'] = pe.block0_conv.out_channels
            if hasattr(pe, 'block1_conv'): cfg['patch_embed']['block1'] = pe.block1_conv.out_channels
            if hasattr(pe, 'block2_conv'): cfg['patch_embed']['block2'] = pe.block2_conv.out_channels
            if hasattr(pe, 'block3_conv'): cfg['patch_embed']['block3'] = pe.block3_conv.out_channels
            if hasattr(pe, 'block4_conv'): cfg['patch_embed']['block4'] = pe.block4_conv.out_channels

        # Blocks
        blocks_cfg = []
        for blk in getattr(self, "block"):
            if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'c_hidden'):
                blocks_cfg.append({'mlp_hidden': blk.mlp.c_hidden})
            else:
                blocks_cfg.append({'mlp_hidden': None})
        cfg['blocks'] = blocks_cfg
        return cfg

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _, hook = patch_embed(x, hook=hook)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            x = x.mean(0)
        return x, hook


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model