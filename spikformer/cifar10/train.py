#!/usr/bin/env python3
# This is a slightly modified version of timm's training script
""" Spikformer ImageNet Testing Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from spikingjelly.clock_driven import functional

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from loader import create_loader
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
import model
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

from Spik4lite import EnergyPenaltyTerm, GumbelTemperatureScheduler, SpikingConv2d, SpikingConv1d
import gc

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='cifar10.yml', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch Classification Training')

# Model detail
parser.add_argument('--model', default='spikformer', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('-T', '--time-step', type=int, default=4, metavar='time',
                    help='simulation time step of spiking neuron (default: 4)')
parser.add_argument('-L', '--layer', type=int, default=4, metavar='layer',
                    help='model layer (default: 4)')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--dim', type=int, default=None, metavar='N',
                    help='embedding dimsension of feature')
parser.add_argument('--num_heads', type=int, default=None, metavar='N',
                    help='attention head number')
parser.add_argument('--patch-size', type=int, default=None, metavar='N',
                    help='Image patch size')
parser.add_argument('--mlp-ratio', type=int, default=None, metavar='N',
                    help='expand ration of embedding dimension in MLP block')
# Dataset / Model parameters
parser.add_argument('-data-dir', metavar='DIR',default="CIFAR10/",
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')

parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N',
                    help='input val batch size for training (default: 32)')
# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[1.0, 1.0], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='spikformer/cifar10/output/train', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

parser.add_argument('--lambda-energy', type=float, default=0.03,
                    help='weight for physics energy loss. '
                         'Higher lambda means more pruning on energy consumption (SOPs).')
parser.add_argument('--gumbel-temp-start', type=float, default=1.0,
                    help='initial Gumbel temperature (default: 1.0)')
parser.add_argument('--gumbel-temp-end', type=float, default=0.5,
                    help='final Gumbel temperature (default: 0.5)')

parser.add_argument('--pruning-interval', type=int, default=30, metavar='N',
                    help='Number of epochs to accumulate mask before pruning (default: 5)')
parser.add_argument('--pruning-start-epoch', type=int, default=20, metavar='N',
                    help='Epoch to start the pruning cycle (default: 10)')
parser.add_argument('--pruning-threshold', type=float, default=0.5, metavar='VAL',
                    help='Threshold for pruning channels based on accumulated mask probability (default: 0.5)')
parser.add_argument('--pruning-end-offset', type=int, default=50, metavar='N',
                    help='Stop pruning this many epochs before the end (fine-tuning phase, default: 20)')

parser.add_argument('--save-start-epoch', type=int, default=0, metavar='N',
                    help='Epoch to start saving checkpoints (default: 0, save from beginning)')
parser.add_argument('--global-pruning', action='store_true', default=True,
                    help='Enable Global Embedding Pruning to prune the residual backbone (default: True)')
parser.add_argument('--pe-max-prune-rate', type=float, default=0.2,
                    help='Maximum pruning rate for Patch Embedding (default: 0.05)')
parser.add_argument('--mlp-max-prune-rate', type=float, default=0.35,
                    help='Maximum pruning rate for MLP blocks (default: 0.2)')

def load_pruned_checkpoint(model, checkpoint_path, log_info=False):
    if log_info:
        _logger.info(f'Loading pruned model from {checkpoint_path}...')
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        _logger.error(f"Failed to load checkpoint file {checkpoint_path}: {e}")
        return None, None, None

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    for name, module in model.named_modules():
        weight_key = f"{name}.weight"
        if weight_key not in state_dict:
            continue
        ckpt_weight = state_dict[weight_key]

        if isinstance(module, (SpikingConv2d, SpikingConv1d)):
            ckpt_out_ch = ckpt_weight.shape[0]
            ckpt_in_ch = ckpt_weight.shape[1] * module.groups
            if ckpt_out_ch != module.out_channels or ckpt_in_ch != module.in_channels:
                if log_info:
                    _logger.info(
                        f"  [Resize] SpikingConv {name}: ({module.in_channels}, {module.out_channels}) -> ({ckpt_in_ch}, {ckpt_out_ch})")
                module.manual_resize(in_channels=ckpt_in_ch, out_channels=ckpt_out_ch)

        elif isinstance(module, nn.Linear):
            ckpt_out = ckpt_weight.shape[0]
            ckpt_in = ckpt_weight.shape[1]
            if ckpt_out != module.out_features or ckpt_in != module.in_features:
                if log_info:
                    _logger.info(
                        f"  [Resize] Linear {name}: ({module.in_features}, {module.out_features}) -> ({ckpt_in}, {ckpt_out})")
                module.in_features = ckpt_in
                module.out_features = ckpt_out
                module.weight = nn.Parameter(torch.empty(ckpt_out, ckpt_in))
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.empty(ckpt_out))

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            ckpt_features = ckpt_weight.shape[0]
            if ckpt_features != module.num_features:
                if log_info:
                    _logger.info(f"  [Resize] BN {name}: {module.num_features} -> {ckpt_features}")
                module.num_features = ckpt_features
                module.weight = nn.Parameter(torch.empty(ckpt_features))
                module.bias = nn.Parameter(torch.empty(ckpt_features))
                module.running_mean = torch.empty(ckpt_features)
                module.running_var = torch.empty(ckpt_features)

        elif isinstance(module, nn.Conv2d):
            ckpt_out_ch = ckpt_weight.shape[0]
            ckpt_in_ch = ckpt_weight.shape[1] * module.groups
            if ckpt_out_ch != module.out_channels or ckpt_in_ch != module.in_channels:
                if log_info:
                    _logger.info(
                        f"  [Resize] Conv2d {name}: ({module.in_channels}, {module.out_channels}) -> ({ckpt_in_ch}, {ckpt_out_ch})")
                module.in_channels = ckpt_in_ch
                module.out_channels = ckpt_out_ch
                module.weight = nn.Parameter(torch.empty(ckpt_out_ch, ckpt_in_ch // module.groups, *module.kernel_size))
                if module.bias is not None:
                    module.bias = nn.Parameter(torch.empty(ckpt_out_ch))

    model.load_state_dict(state_dict, strict=False)
    if log_info:
        _logger.info("Successfully loaded pruned weights into resized model.")

    return checkpoint.get('epoch', None), checkpoint.get('optimizer', None), checkpoint.get('amp_scaler', None)

def set_model_mask_accumulation(model, enable=True):
    real_model = model.module if hasattr(model, 'module') else model
    for m in real_model.modules():
        if isinstance(m, (SpikingConv2d, SpikingConv1d)):
            if enable:
                if not m.enable_mask_accumulation:
                    m.start_mask_accumulation()
            else:
                m.enable_mask_accumulation = False

def update_timm_saver(saver, model, optimizer, model_ema=None):
    if saver is not None:
        saver.model = model
        saver.optimizer = optimizer
        if model_ema is not None:
            saver.model_ema = model_ema

class CustomCheckpointSaver(CheckpointSaver):
    def save_checkpoint(self, epoch, metric=None, structure_config=None):

        self.structure_config_to_save = structure_config

        return super().save_checkpoint(epoch, metric)

    def _save(self, save_path, epoch, metric=None):

        super()._save(save_path, epoch, metric)

        if hasattr(self, 'structure_config_to_save') and self.structure_config_to_save is not None:
            try:
                ckpt = torch.load(save_path, map_location='cpu')
                ckpt['structure_config'] = self.structure_config_to_save
                torch.save(ckpt, save_path)
            except Exception as e:
                _logger.warning(f"Failed to append structure_config to checkpoint: {e}")

            self.structure_config_to_save = None

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:1'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)


    pruned_cfg = None
    if args.resume and os.path.exists(args.resume):
        try:

            temp_ckpt = torch.load(args.resume, map_location='cpu')
            if 'structure_config' in temp_ckpt:
                pruned_cfg = temp_ckpt['structure_config']
                if args.local_rank == 0:
                    _logger.info("Found 'structure_config' in checkpoint! Initializing model with pruned structure.")
            del temp_ckpt
        except Exception as e:
            if args.local_rank == 0: _logger.warning(f"Error checking checkpoint: {e}")

    model = create_model(
        'spikformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=args.patch_size, embed_dims=args.dim, num_heads=args.num_heads, mlp_ratios=args.mlp_ratio,
        in_channels=3, num_classes=args.num_classes, qkv_bias=False,
        depths=args.layer, sr_ratios=1,
        pruned_structure_cfg=pruned_cfg 
    )
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:

        resume_epoch, loaded_optimizer_state, loaded_scaler_state = load_pruned_checkpoint(
            model, args.resume, log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    # setup distributed training
    if args.distributed:
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank],find_unused_parameters=True)  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir, split=args.train_split, is_training=True,
        batch_size=args.batch_size, repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.val_batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    regularizer_criterion = None



    
    if args.local_rank == 0:
        _logger.info(f"Using Physics-Aware Mode: Minimizing Energy (SOPs). Lambda={args.lambda_energy}")

    regularizer_criterion = EnergyPenaltyTerm(
        model,
        lambda_energy=args.lambda_energy
    )

    temp_scheduler = GumbelTemperatureScheduler(
        model,
        init_temp=getattr(args, 'gumbel_temp_start', 5.0),
        final_temp=getattr(args, 'gumbel_temp_end', 0.1),
        total_epochs=args.epochs
    )

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CustomCheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing,
            #max_history=args.checkpoint_hist
            max_history=1)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    fine_tuning_start_epoch = args.epochs - args.pruning_end_offset

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            is_accumulating_phase = False
            is_pruning_epoch = False

            if args.pruning_start_epoch <= epoch < (args.epochs - args.pruning_end_offset):

                cycle_index = (epoch - args.pruning_start_epoch) % args.pruning_interval

                is_accumulating_phase = True

                if cycle_index == (args.pruning_interval - 1):
                    is_pruning_epoch = True

            set_model_mask_accumulation(model, is_accumulating_phase)

            if epoch == fine_tuning_start_epoch:
                if args.local_rank == 0:
                    _logger.info("****************************************************************")
                    _logger.info(f"*** Epoch {epoch}: Transitioning to FINE-TUNING Phase ***")
                    _logger.info("*** Permanently removing Gating Layers from model... ***")
                    _logger.info("****************************************************************")

                real_model = model.module if hasattr(model, 'module') else model

                removed_count = 0
                for m in real_model.modules():
                    if isinstance(m, (SpikingConv2d, SpikingConv1d)):
                        if hasattr(m, 'gating_layer'):
                            del m.gating_layer
                            m.gating_layer = None
                        if hasattr(m, 'running_fr'): del m.running_fr
                        if hasattr(m, 'current_probs'): m.current_probs = None
                        m.static_mode = True
                        removed_count += 1

                if args.local_rank == 0:
                    _logger.info(f"Removed gating from {removed_count} layers. Model is now Clean SNN.")

                gc.collect()
                torch.cuda.empty_cache()

                if args.local_rank == 0:
                    _logger.info("Cleaning up optimizer params instead of rebuilding to preserve momentum...")

                valid_params = set(real_model.parameters())

                for group in optimizer.param_groups:
                    new_params = [p for p in group['params'] if p in valid_params]
                    group['params'] = new_params

                keys_to_remove = [p for p in optimizer.state.keys() if p not in valid_params]
                for p in keys_to_remove:
                    del optimizer.state[p]

                if args.local_rank == 0:
                    _logger.info(f"Optimizer clean-up finished. Removed {len(keys_to_remove)} expired params.")

                if args.distributed:
                    if args.local_rank == 0: _logger.info("Re-wrapping DDP...")
                    if has_apex and args.apex_amp:
                        model = ApexDDP(real_model, delay_allreduce=True)
                    else:
                        model = NativeDDP(real_model, device_ids=[args.local_rank])

                if model_ema is not None:
                    model_ema = ModelEmaV2(model, decay=args.model_ema_decay,
                                           device='cpu' if args.model_ema_force_cpu else None)

                update_timm_saver(saver, model, optimizer, model_ema)

            if args.local_rank == 0:
                phase_str = "Accumulating" if is_accumulating_phase else "Normal"
                if is_pruning_epoch: phase_str += " -> Pruning Next"
                if epoch >= fine_tuning_start_epoch: phase_str = "FINE-TUNING (No Gating)"
                _logger.info(f"Epoch {epoch} Phase: {phase_str}")

            current_regularizer = regularizer_criterion
            if epoch >= fine_tuning_start_epoch:
                current_regularizer = None

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn,
                regularizer_fn=current_regularizer,
                temp_scheduler=temp_scheduler
            )

            if is_pruning_epoch:
                if args.local_rank == 0:
                    _logger.info(
                        f"*** [Epoch {epoch}] Executing Physical Pruning (Average over {args.pruning_interval} epochs) ***")

                    if args.global_pruning:
                        _logger.info("*** Global Embedding Pruning is ENABLED ***")

                real_model = model.module if hasattr(model, 'module') else model
                prev_params = sum(p.numel() for p in real_model.parameters())

                torch.cuda.empty_cache()

                if hasattr(real_model, 'prune_model'):

                    pruning_stats = real_model.prune_model(
                        threshold=args.pruning_threshold,
                        optimizer=optimizer,
                        global_pruning=args.global_pruning,
                        pe_max_prune_rate=args.pe_max_prune_rate,
                        mlp_max_prune_rate=args.mlp_max_prune_rate
                    )

                curr_params = sum(p.numel() for p in real_model.parameters())
                if args.local_rank == 0:
                    _logger.info(f"  Params: {prev_params} -> {curr_params}")

                if curr_params != prev_params:

                    optimizer.zero_grad(set_to_none=True)

                    if args.local_rank == 0:
                        _logger.info("Optimizer state preserved via surgical pruning (not rebuilt).")

                    if has_apex and args.apex_amp:
                        # Apex state cleanup if needed
                        pass

                    gc.collect()
                    torch.cuda.empty_cache()

                    if args.distributed:
                        if args.local_rank == 0: _logger.info("Re-wrapping DistributedDataParallel.")
                        if has_apex and args.apex_amp:
                            model = ApexDDP(real_model, delay_allreduce=True)
                        else:
                            model = NativeDDP(real_model, device_ids=[args.local_rank])

                    if model_ema is not None:
                        if args.local_rank == 0: _logger.warning("Resetting Model EMA due to pruning.")
                        model_ema = ModelEmaV2(model, decay=args.model_ema_decay,
                                               device='cpu' if args.model_ema_force_cpu else None)

                    update_timm_saver(saver, model, optimizer, model_ema)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]

                real_model_to_save = model.module if hasattr(model, 'module') else model
                current_structure_cfg = None
                if hasattr(real_model_to_save, 'export_structure_config'):
                    current_structure_cfg = real_model_to_save.export_structure_config()

                if epoch >= args.save_start_epoch:
                    best_metric, best_epoch = saver.save_checkpoint(
                        epoch, metric=save_metric,

                        structure_config=current_structure_cfg
                    )
                    _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None,

        regularizer_fn=None, temp_scheduler=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()


    gating_losses_m = AverageMeter()
    temp_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss_cls = loss_fn(output, target)

            loss_reg = torch.tensor(0.0, device=input.device)

            if regularizer_fn is not None:

                loss_reg = regularizer_fn()

            loss = loss_cls + loss_reg

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            gating_losses_m.update(loss_reg.item(), input.size(0))

            if temp_scheduler:
                temp_m.update(temp_scheduler.get_temp(), 1)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            # loss.backward()
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        functional.reset_net(model)

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)

                reduced_reg_loss = reduce_tensor(loss_reg.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                gating_losses_m.update(reduced_reg_loss.item(), input.size(0))

            if args.local_rank == 0:

                current_temp = temp_scheduler.get_temp() if temp_scheduler else 0.0
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'RegLoss: {reg_loss.val:>7.4f} ({reg_loss.avg:>6.4f})  '
                    'Temp: {temp:.4f}  '
                    'Time: {batch_time.val:.3f}s  '
                    'LR: {lr:.3e}'.format(
                        epoch, batch_idx, len(loader), 100. * batch_idx / last_idx,
                        loss=losses_m,
                        reg_loss=gating_losses_m,
                        temp=current_temp,
                        batch_time=batch_time_m,
                        lr=lr))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    if temp_scheduler is not None:
        temp_scheduler.step()

    return OrderedDict([('loss', losses_m.avg), ('gating_loss', gating_losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()
