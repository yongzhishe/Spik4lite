import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import to_2tuple

from module.Spik4lite import SpikingConv2d, prune_batchnorm

class MS_SPS(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=4,
            in_channels=2,
            embed_dims=256,
            pooling_stat="1111",
            spike_mode="lif",
            custom_dims=None                             
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.pooling_stat = pooling_stat
        
        self.C = in_channels
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        self.num_patches = self.H * self.W
        dims = {
            'block0': embed_dims // 8,
            'block1': embed_dims // 4,
            'block2': embed_dims // 2,
            'block3': embed_dims,
            'block4': embed_dims
        }

        if custom_dims is not None:
            dims.update(custom_dims)

        self.proj_conv = nn.Conv2d(
            in_channels, dims['block0'], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(dims['block0'])
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv1 = SpikingConv2d(
            dims['block0'],
            dims['block1'],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn1 = nn.BatchNorm2d(dims['block1'])
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = SpikingConv2d(
            dims['block1'],
            dims['block2'],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.proj_bn2 = nn.BatchNorm2d(dims['block2'])
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = SpikingConv2d(
            dims['block2'], dims['block3'], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(dims['block3'])
        if spike_mode == "lif":
            self.proj_lif3 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.proj_lif3 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = SpikingConv2d(
            dims['block3'], dims['block4'], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(dims['block4'])
        if spike_mode == "lif":
            self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.rpe_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )
        self.block0_conv = self.proj_conv
        self.block1_conv = self.proj_conv1
        self.block2_conv = self.proj_conv2
        self.block3_conv = self.proj_conv3
        self.block4_conv = self.rpe_conv
        self.block4_bn = self.rpe_bn
        
    def forward(self, x, hook=None):
        T, B, _, H, W = x.shape
        ratio = 1

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()

        if self.pooling_stat[0] == "1":
            x = x.flatten(0, 1).contiguous()
            x = self.maxpool(x)
            ratio *= 2
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x.flatten(0, 1)).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()

        if self.pooling_stat[1] == "1":
            x = x.flatten(0, 1).contiguous()
            x = self.maxpool1(x)
            ratio *= 2
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x.flatten(0, 1)).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()

        if self.pooling_stat[2] == "1":
            x = x.flatten(0, 1).contiguous()
            x = self.maxpool2(x)
            ratio *= 2
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        x = self.proj_conv3(x)
        x = self.proj_bn3(x.flatten(0, 1)).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        if self.pooling_stat[3] == "1":
            x = x.flatten(0, 1).contiguous()
            x = self.maxpool3(x)
            ratio *= 2
            x = x.reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        x_feat = x
        
        x = self.proj_lif3(x) 
        if hook is not None:
            hook[self._get_name() + "_lif3"] = x.detach()

        x = self.rpe_conv(x)
        x = self.rpe_bn(x.flatten(0, 1)).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        x = x + x_feat

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook

    def _get_keep_indices(self, probs, current_dim, threshold, max_prune_rate):
        if probs is None:
            return None

        sorted_scores, sorted_indices = torch.sort(probs, descending=True)

        keep_mask = sorted_scores >= threshold
        num_to_keep = keep_mask.sum().item()

        min_keep = int(current_dim * (1.0 - max_prune_rate))

        num_to_keep = max(num_to_keep, min_keep, 8)

        keep_indices = sorted_indices[:num_to_keep]
        keep_indices, _ = keep_indices.sort()
        
        return keep_indices

    def prune_parameters(self, threshold=0.3, optimizer=None, max_prune_rate=0.1):
        stats = []

        if isinstance(self.proj_conv2, SpikingConv2d) and isinstance(self.proj_conv1, SpikingConv2d):

            probs = self.proj_conv2.get_average_mask_probs()

            total_channels = self.proj_conv2.in_channels 

            keep_idx = self._get_keep_indices(probs, total_channels, threshold, max_prune_rate)
            
            if keep_idx is not None and len(keep_idx) < total_channels:
                kept = len(keep_idx)

                self.proj_conv2.prune_in_channels(keep_idx, optimizer)

                prune_batchnorm(self.proj_bn1, keep_idx, optimizer)

                self.proj_conv1.prune_out_channels(keep_idx, optimizer)
                
                stats.append(('block1_out', total_channels, kept))

        if isinstance(self.proj_conv3, SpikingConv2d) and isinstance(self.proj_conv2, SpikingConv2d):

            probs = self.proj_conv3.get_average_mask_probs()
            total_channels = self.proj_conv3.in_channels

            keep_idx = self._get_keep_indices(probs, total_channels, threshold, max_prune_rate)
            
            if keep_idx is not None and len(keep_idx) < total_channels:
                kept = len(keep_idx)

                self.proj_conv3.prune_in_channels(keep_idx, optimizer)

                prune_batchnorm(self.proj_bn2, keep_idx, optimizer)

                self.proj_conv2.prune_out_channels(keep_idx, optimizer)
                
                stats.append(('block2_out', total_channels, kept))

        
        return stats