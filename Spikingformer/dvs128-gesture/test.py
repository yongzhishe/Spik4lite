import datetime
import os
import time
import logging
import gc
import yaml

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp

import model, utils
from Spik4lite import SpikingConv2d
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import dvs128_gesture
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import *
import autoaugment

_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
writer = SummaryWriter("./")
def infer_structure_config_from_state_dict(state_dict):

    cfg = {}

    pe_cfg = {}
    for i in range(5):
        key = f'patch_embed.block{i}_conv.weight'
        if key in state_dict:
            pe_cfg[f'block{i}'] = state_dict[key].shape[0]
    if pe_cfg:
        cfg['patch_embed'] = pe_cfg

    blocks_cfg = []
    i = 0
    while True:
        key = f'block.{i}.mlp.mlp1_conv.weight'
        if key not in state_dict:
            break
        hidden_dim = state_dict[key].shape[0]
        blocks_cfg.append({'mlp_hidden': hidden_dim})
        i += 1
    if blocks_cfg:
        cfg['blocks'] = blocks_cfg
    return cfg

def get_pruned_config(checkpoint_path):

    if not os.path.exists(checkpoint_path):
        return None, 0
    try:
        print(f"Peeking at checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        epoch = ckpt.get('epoch', 0) if isinstance(ckpt, dict) else 0

        if isinstance(ckpt, dict) and 'structure_config' in ckpt:
            print("Found explicit 'structure_config' in checkpoint.")
            return ckpt['structure_config'], epoch

        print("No explicit config found. Inferring structure from weight shapes...")
        state_dict = ckpt
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']

        inferred_cfg = infer_structure_config_from_state_dict(state_dict)
        return inferred_cfg, epoch
    except Exception as e:
        print(f"Error reading checkpoint for config: {e}")
        return None, 0

def clean_and_set_inference_mode(model):

    print("Cleaning up model: removing gating layers and setting inference mode...")
    removed_count = 0
    for m in model.modules():
        if isinstance(m, SpikingConv2d):
            if hasattr(m, 'gating_layer'):
                del m.gating_layer
                m.gating_layer = None
                removed_count += 1
            if hasattr(m, 'running_fr'):
                del m.running_fr
            if hasattr(m, 'current_probs'): m.current_probs = None
            if hasattr(m, 'current_cost_coeff'): m.current_cost_coeff = None
            if hasattr(m, 'mask_accumulator'): m.mask_accumulator = None
            m.static_mode = True
            if hasattr(m, 'static_mask'): m.static_mask = None
            
    print(f"Removed gating layers from {removed_count} SpikingConv2d modules.")

def load_weights_ignoring_gating(model, checkpoint_path):

    print(f"Loading weights from {checkpoint_path} (Filtering gating keys)...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():

        if 'gating_layer' in k or 'running_fr' in k or 'static_mask' in k:
            continue
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if len(missing) > 0:
        real_missing = [k for k in missing if 'gating_layer' not in k and 'running_fr' not in k]
        if real_missing:
            print(f"Warning: Missing keys (excluding gating): {real_missing}")
            
    print("Weights loaded successfully.")
    return checkpoint


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='SEWResNet', help='model')
    parser.add_argument('--dataset', default='DVS128Gesture', help='dataset')
    parser.add_argument('--num-classes', type=int, default=11, metavar='N',
                        help='number of label classes (default: 1000)')
    parser.add_argument('--data-path', default='DVS128Gesture/', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='spikingformer/DVS128Gesture/output/test', help='path where to save')
    parser.add_argument('--resume', default='checkpoint_max_test_acc1.pth', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
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
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=192, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    args = parser.parse_args()
    return args

_logger = logging.getLogger("test")
stream_handler = logging.StreamHandler()
format_str = "%(asctime)s %(levelname)s: %(message)s"
stream_handler.setFormatter(logging.Formatter(format_str))
_logger.addHandler(stream_handler)
_logger.propagate = False

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    end = time.time()
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

            metric_logger.meters['batch_time'].update(time.time() - end)
            end = time.time()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5, batch_time = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg, metric_logger.batch_time.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}, time = {batch_time}')

    mem_summary = torch.cuda.memory_summary(device=0, abbreviated=False)
    
    print("CUDA Memory Summary:\n" + mem_summary)

    return loss, acc1, acc5

def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T,
                                                 split_by='number')
    dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                split_by='number')
    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'


    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)

    data_path = args.data_path

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True)

    pruned_cfg = None
    if args.resume:
        pruned_cfg, _ = get_pruned_config(args.resume)

    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
        pruned_structure_cfg=pruned_cfg
    )

    clean_and_set_inference_mode(model)

    print("Creating model done")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)
    
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    start_epoch = 0
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:

        gc.collect()
        torch.cuda.empty_cache()

        checkpoint = load_weights_ignoring_gating(model_without_ddp, args.resume)

        if not args.test_only:

             try:
                 optimizer.load_state_dict(checkpoint['optimizer'])
                 lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
             except Exception as e:
                 print(f"Warning: Failed to load optimizer state: {e}. Starting optimizer from scratch.")
                 
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
        if 'max_test_acc1' in checkpoint:
            max_test_acc1 = checkpoint['max_test_acc1']

        gc.collect()
        torch.cuda.empty_cache()


    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))
        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    train_snn_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    print("Start testing")
    start_time = time.time()
    
    with torch.no_grad(): 
        for epoch in range(args.start_epoch, num_epochs):
            save_max = False
            if args.distributed:
                train_sampler.set_epoch(epoch)
            if epoch >= 75:
                mixup_fn.mixup_enabled = False
            
            
            test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
            
            if te_tb_writer is not None:
                if utils.is_main_process():
                    te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                    te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                    te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

            if max_test_acc1 < test_acc1:
                max_test_acc1 = test_acc1
                test_acc5_at_max_test_acc1 = test_acc5
                save_max = True

            if output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_test_acc1': max_test_acc1,
                    'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                }
                if save_max:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
            print(output_dir)
            
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1

if __name__ == "__main__":
    args = parse_args()
    main(args)