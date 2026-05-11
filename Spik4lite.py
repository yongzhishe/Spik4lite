import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.common_types import _size_2_t

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

            if value.dim() > pruning_dim:

                if value.shape[pruning_dim] == old_param.shape[pruning_dim]:

                    indices_on_device = keep_indices.to(value.device)
                    new_state[key] = value.index_select(pruning_dim, indices_on_device).detach()
                else:
                    new_state[key] = value.detach()
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

class SpikingConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 temperature=5.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)

        self.gating_layer = nn.Linear(in_channels, in_channels * 2)

        self.gating_layer.bias.data = torch.tensor([0.5, 2.0]).repeat(in_channels)
        self.temperature = temperature

        self.register_buffer('running_fr', torch.zeros(in_channels))
        self.momentum = 0.9
        self.static_flop_cost = None
        self.current_cost_coeff = None
        self.current_probs = None

        self.mask_accumulator = None
        self.accumulation_steps = 0
        self.enable_mask_accumulation = False

        self.static_mode = False

    def set_temperature(self, temp):
        self.temperature = temp

    def start_mask_accumulation(self):
        self.enable_mask_accumulation = True
        self.mask_accumulator = None
        self.accumulation_steps = 0

    def get_average_mask_probs(self):
        if self.mask_accumulator is None or self.accumulation_steps == 0:
            return None
        avg_probs = self.mask_accumulator / self.accumulation_steps
        self.mask_accumulator = None
        self.accumulation_steps = 0
        self.enable_mask_accumulation = False
        return avg_probs

    def prune_in_channels(self, keep_indices, optimizer=None):
        keep_indices = keep_indices.to(self.weight.device)
        new_in_channels = len(keep_indices)

        with torch.no_grad():

            old_weight = self.weight
            new_weight = nn.Parameter(self.weight.data[:, keep_indices, :])

            if optimizer is not None:
                prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)

            self.weight = new_weight
            self.in_channels = new_in_channels
            self.mask_accumulator = None
            self.accumulation_steps = 0

            if hasattr(self, 'running_fr'):
                self.running_fr = self.running_fr[keep_indices]

        if hasattr(self, 'gating_layer') and self.gating_layer is not None:
            old_gating = self.gating_layer
            device = old_gating.weight.device

            out_keep_indices = []
            for idx in keep_indices:
                idx_val = idx.item()
                out_keep_indices.append(2 * idx_val)
                out_keep_indices.append(2 * idx_val + 1)
            out_keep_indices = torch.tensor(out_keep_indices, device=device, dtype=torch.int)

            new_gating = nn.Linear(new_in_channels, new_in_channels * 2).to(device)
            
            with torch.no_grad():
                temp_weight = old_gating.weight.index_select(1, keep_indices)
                new_weight = temp_weight.index_select(0, out_keep_indices)
                new_bias = old_gating.bias.index_select(0, out_keep_indices)
                new_gating.weight.copy_(new_weight)
                new_gating.bias.copy_(new_bias)

            if optimizer is not None and old_gating.weight in optimizer.state:
                old_w_param = old_gating.weight
                new_w_param = new_gating.weight
                old_w_state = optimizer.state[old_w_param]
                new_w_state = {}

                for key, value in old_w_state.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 0:

                        if value.shape[1] == old_w_param.shape[1]:
                            temp_val = value.index_select(1, keep_indices)
                        else:
                            temp_val = value

                        if value.shape[0] == old_w_param.shape[0]:
                            final_val = temp_val.index_select(0, out_keep_indices).detach()
                        else:
                            final_val = temp_val.detach()
                        new_w_state[key] = final_val
                    else:
                        new_w_state[key] = value
                
                optimizer.state[new_w_param] = new_w_state
                del optimizer.state[old_w_param]

                if old_gating.bias is not None:
                    old_b_param = old_gating.bias
                    new_b_param = new_gating.bias
                    old_b_state = optimizer.state[old_b_param]
                    new_b_state = {}
                    for key, value in old_b_state.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 0:
                            if value.shape[0] == old_b_param.shape[0]:
                                new_b_state[key] = value.index_select(0, out_keep_indices).detach()
                            else:
                                new_b_state[key] = value.detach()
                        else:
                            new_b_state[key] = value
                    optimizer.state[new_b_param] = new_b_state
                    del optimizer.state[old_b_param]

                for group in optimizer.param_groups:
                    for i, p in enumerate(group['params']):
                        if p is old_w_param: group['params'][i] = new_w_param
                        elif old_gating.bias is not None and p is old_gating.bias: group['params'][i] = new_gating.bias

            # Fallback
            elif optimizer is not None:
                 for p in old_gating.parameters():
                    if p in optimizer.state: del optimizer.state[p]
                    for group in optimizer.param_groups:
                         for i, existing_p in enumerate(group['params']):
                            if existing_p is p:
                                group['params'].pop(i)
                                break
                 if len(optimizer.param_groups) > 0:
                     optimizer.param_groups[0]['params'].extend(list(new_gating.parameters()))

            self.gating_layer = new_gating

        self.static_flop_cost = None

    def prune_out_channels(self, keep_indices, optimizer=None):
        keep_indices = keep_indices.to(self.weight.device)
        new_out_channels = len(keep_indices)

        with torch.no_grad():

            old_weight = self.weight
            new_weight = nn.Parameter(self.weight.data[keep_indices, :, :])

            if optimizer is not None:
                prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)

            self.weight = new_weight
            self.mask_accumulator = None
            self.accumulation_steps = 0

            if self.bias is not None:
                old_bias = self.bias
                new_bias = nn.Parameter(self.bias.data[keep_indices])
                if optimizer is not None:
                    prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)
                self.bias = new_bias

            self.out_channels = new_out_channels

        self.static_flop_cost = None

    def manual_resize(self, in_channels=None, out_channels=None):
        if in_channels is not None and in_channels != self.in_channels:
            self.in_channels = in_channels
            new_weight_shape = list(self.weight.shape)
            new_weight_shape[1] = in_channels
            self.weight = nn.Parameter(torch.empty(*new_weight_shape, device=self.weight.device))
            if hasattr(self, 'running_fr'):
                self.register_buffer('running_fr', torch.zeros(in_channels, device=self.weight.device))
            if hasattr(self, 'gating_layer'):
                self.gating_layer = nn.Linear(in_channels, in_channels * 2).to(self.weight.device)

        if out_channels is not None and out_channels != self.out_channels:
            self.out_channels = out_channels
            new_weight_shape = list(self.weight.shape)
            new_weight_shape[0] = out_channels
            self.weight = nn.Parameter(torch.empty(*new_weight_shape, device=self.weight.device))
            if self.bias is not None:
                self.bias = nn.Parameter(torch.empty(out_channels, device=self.bias.device))
        
        self.static_flop_cost = None

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if getattr(self, 'static_mode', False):
            s += ', static_mode=True'
        return s.format(**self.__dict__)

    def forward(self, x):
        T, B, C, L = x.shape
        L_out = (L + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1

        if getattr(self, 'static_mode', False) or getattr(self, 'gating_layer', None) is None:
            x_flat = x.flatten(0, 1) # (T*B, C, L)
            out_flat = F.conv1d(x_flat, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            return out_flat.view(T, B, self.out_channels, L_out)

        current_fr = x.mean(dim=(0, 3)) 
        avg_batch_fr = current_fr.mean(dim=0)

        if self.training:
            if self.running_fr.sum() == 0:
                self.running_fr.data.copy_(avg_batch_fr.detach())
            else:
                self.running_fr = self.momentum * self.running_fr + \
                                  (1 - self.momentum) * avg_batch_fr.detach()
            fr_basis = self.running_fr
        else:
            fr_basis = avg_batch_fr

        # 2. FLOPs Cost
        if self.static_flop_cost is None:
            # 1D FLOPs: Kernel_Size * Out_Channels * Output_Length
            cost_val = self.kernel_size[0] * self.out_channels * L_out
            self.static_flop_cost = cost_val / 1e6

        self.current_cost_coeff = (self.static_flop_cost * fr_basis).detach()

        # 3. Gumbel Softmax
        logits = self.gating_layer(current_fr).view(B, C, 2)

        if self.training:
            y_hard = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            probs = F.softmax(logits, dim=-1)
            if self.enable_mask_accumulation:
                batch_avg_prob = y_hard[:, :, 1].mean(dim=0).detach()
                if self.mask_accumulator is None:
                    self.mask_accumulator = batch_avg_prob
                else:
                    self.mask_accumulator += batch_avg_prob
                self.accumulation_steps += 1
        else:
            indices = logits.argmax(dim=-1)
            y_hard = F.one_hot(indices, num_classes=2).float()
            probs = F.softmax(logits, dim=-1)

        self.current_probs = probs

        mask = y_hard[:, :, 1]
        x_masked = x * mask.view(1, B, C, 1)

        x_flat = x_masked.flatten(0, 1)
        out_flat = F.conv1d(x_flat, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        _, C_out, _ = out_flat.shape
        return out_flat.view(T, B, C_out, L_out)


class SpikingConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 temperature=5.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)

        self.gating_layer = nn.Linear(in_channels, in_channels * 2)
        self.gating_layer.bias.data = torch.tensor([0.5, 2.0]).repeat(in_channels)

        self.temperature = temperature

        self.register_buffer('running_fr', torch.zeros(in_channels))
        self.momentum = 0.9

        self.static_flop_cost = None

        self.current_cost_coeff = None
        self.current_probs = None

        self.mask_accumulator = None
        self.accumulation_steps = 0
        self.enable_mask_accumulation = False

        self.static_mode = False

    def set_temperature(self, temp):
        self.temperature = temp

    def start_mask_accumulation(self):
        self.enable_mask_accumulation = True
        self.mask_accumulator = None
        self.accumulation_steps = 0

    def get_average_mask_probs(self):
        if self.mask_accumulator is None or self.accumulation_steps == 0:
            return None
        avg_probs = self.mask_accumulator / self.accumulation_steps
        self.mask_accumulator = None
        self.accumulation_steps = 0
        self.enable_mask_accumulation = False
        return avg_probs

    def prune_in_channels(self, keep_indices, optimizer=None):

        keep_indices = keep_indices.to(self.weight.device)
        new_in_channels = len(keep_indices)

        with torch.no_grad():

            old_weight = self.weight
            new_weight = nn.Parameter(self.weight.data[:, keep_indices, :, :])

            if optimizer is not None:
                prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=1)

            self.weight = new_weight
            self.in_channels = new_in_channels
            self.mask_accumulator = None
            self.accumulation_steps = 0

            if hasattr(self, 'running_fr'):
                self.running_fr = self.running_fr[keep_indices]

        if hasattr(self, 'gating_layer') and self.gating_layer is not None:
            old_gating = self.gating_layer
            device = old_gating.weight.device

            out_keep_indices = []
            for idx in keep_indices:
                idx_val = idx.item()
                out_keep_indices.append(2 * idx_val)  # Drop neuron
                out_keep_indices.append(2 * idx_val + 1)  # Keep neuron

            out_keep_indices = torch.tensor(out_keep_indices, device=device, dtype=torch.int)

            new_gating = nn.Linear(new_in_channels, new_in_channels * 2).to(device)

            with torch.no_grad():

                temp_weight = old_gating.weight.index_select(1, keep_indices)

                new_weight = temp_weight.index_select(0, out_keep_indices)

                new_bias = old_gating.bias.index_select(0, out_keep_indices)

                new_gating.weight.copy_(new_weight)
                new_gating.bias.copy_(new_bias)

            if optimizer is not None and old_gating.weight in optimizer.state:
                old_w_param = old_gating.weight
                new_w_param = new_gating.weight

                old_w_state = optimizer.state[old_w_param]
                new_w_state = {}

                for key, value in old_w_state.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 0:
                        if value.shape[1] == old_w_param.shape[1]:
                            temp_val = value.index_select(1, keep_indices)
                        else:
                            temp_val = value

                        if value.shape[0] == old_w_param.shape[0]:
                            final_val = temp_val.index_select(0, out_keep_indices).detach()
                        else:
                            final_val = temp_val.detach()

                        new_w_state[key] = final_val
                    else:
                        new_w_state[key] = value

                optimizer.state[new_w_param] = new_w_state
                del optimizer.state[old_w_param]

                if old_gating.bias is not None:
                    old_b_param = old_gating.bias
                    new_b_param = new_gating.bias

                    old_b_state = optimizer.state[old_b_param]
                    new_b_state = {}

                    for key, value in old_b_state.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 0:
                            if value.shape[0] == old_b_param.shape[0]:
                                new_b_state[key] = value.index_select(0, out_keep_indices).detach()
                            else:
                                new_b_state[key] = value.detach()
                        else:
                            new_b_state[key] = value

                    optimizer.state[new_b_param] = new_b_state
                    del optimizer.state[old_b_param]

                for group in optimizer.param_groups:
                    for i, p in enumerate(group['params']):
                        if p is old_w_param:
                            group['params'][i] = new_w_param
                        elif old_gating.bias is not None and p is old_gating.bias:
                            group['params'][i] = new_gating.bias

            elif optimizer is not None:

                for p in old_gating.parameters():
                    if p in optimizer.state: del optimizer.state[p]
                    for group in optimizer.param_groups:
                        for i, existing_p in enumerate(group['params']):
                            if existing_p is p:
                                group['params'].pop(i)
                                break
                if len(optimizer.param_groups) > 0:
                    optimizer.param_groups[0]['params'].extend(list(new_gating.parameters()))

            self.gating_layer = new_gating

        self.static_flop_cost = None

    def prune_out_channels(self, keep_indices, optimizer=None):

        keep_indices = keep_indices.to(self.weight.device)
        new_out_channels = len(keep_indices)

        with torch.no_grad():

            old_weight = self.weight
            new_weight = nn.Parameter(self.weight.data[keep_indices, :, :, :])

            if optimizer is not None:
                prune_optimizer_state(optimizer, old_weight, new_weight, keep_indices, pruning_dim=0)

            self.weight = new_weight
            self.mask_accumulator = None
            self.accumulation_steps = 0

            if self.bias is not None:

                old_bias = self.bias
                new_bias = nn.Parameter(self.bias.data[keep_indices])

                if optimizer is not None:
                    prune_optimizer_state(optimizer, old_bias, new_bias, keep_indices, pruning_dim=0)

                self.bias = new_bias

            self.out_channels = new_out_channels

        self.static_flop_cost = None

    def manual_resize(self, in_channels=None, out_channels=None):
 
        if in_channels is not None:

            if in_channels != self.in_channels:
                self.in_channels = in_channels

                new_weight_shape = list(self.weight.shape)
                new_weight_shape[1] = in_channels
                self.weight = nn.Parameter(torch.empty(*new_weight_shape, device=self.weight.device))

                if hasattr(self, 'running_fr'):
                    self.register_buffer('running_fr', torch.zeros(in_channels, device=self.weight.device))

                if hasattr(self, 'gating_layer'):
                    self.gating_layer = nn.Linear(in_channels, in_channels * 2).to(self.weight.device)

        if out_channels is not None:
            if out_channels != self.out_channels:
                self.out_channels = out_channels
                new_weight_shape = list(self.weight.shape)
                new_weight_shape[0] = out_channels
                self.weight = nn.Parameter(torch.empty(*new_weight_shape, device=self.weight.device))
                if self.bias is not None:
                    self.bias = nn.Parameter(torch.empty(out_channels, device=self.bias.device))

        self.static_flop_cost = None

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if getattr(self, 'static_mode', False):
            s += ', static_mode=True'
        return s.format(**self.__dict__)

    def forward(self, x):
        T, B, C, H, W = x.shape

        h_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        if getattr(self, 'static_mode', False) or getattr(self, 'gating_layer', None) is None:
            x_flat = x.flatten(0, 1)
            out_flat = F.conv2d(x_flat, self.weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
            return out_flat.view(T, B, self.out_channels, h_out, w_out)

        current_fr = x.mean(dim=(0, 3, 4))
        avg_batch_fr = current_fr.mean(dim=0)

        if self.training:

            if self.running_fr.sum() == 0:
                self.running_fr.data.copy_(avg_batch_fr.detach())
            else:
                self.running_fr = self.momentum * self.running_fr + \
                                  (1 - self.momentum) * avg_batch_fr.detach()
            fr_basis = self.running_fr
        else:
            fr_basis = avg_batch_fr

        if self.static_flop_cost is None:
            cost_val = self.kernel_size[0] * self.kernel_size[1] * self.out_channels * h_out * w_out
            self.static_flop_cost = cost_val / 1e6

        self.current_cost_coeff = (self.static_flop_cost * fr_basis).detach()

        logits = self.gating_layer(current_fr).view(B, C, 2)

        if self.training:
            y_hard = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            probs = F.softmax(logits, dim=-1)

            if self.enable_mask_accumulation:

                batch_avg_prob = y_hard[:, :, 1].mean(dim=0).detach()

                if self.mask_accumulator is None:
                    self.mask_accumulator = batch_avg_prob
                else:
                    self.mask_accumulator += batch_avg_prob
                self.accumulation_steps += 1
        else:
            indices = logits.argmax(dim=-1)
            y_hard = F.one_hot(indices, num_classes=2).float()
            probs = F.softmax(logits, dim=-1)

        self.current_probs = probs

        # Masking
        mask = y_hard[:, :, 1]

        x_masked = x * mask.view(1, B, C, 1, 1)

        # Conv Forward
        x_flat = x_masked.flatten(0, 1)
        out_flat = F.conv2d(x_flat, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        _, C_out, _, _ = out_flat.shape
        return out_flat.view(T, B, C_out, h_out, w_out)


class EnergyPenaltyTerm(nn.Module):
    def __init__(self, model: nn.Module, lambda_energy: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.lambda_energy = lambda_energy
        self.conv_layers = []
        for m in model.modules():
            if isinstance(m, (SpikingConv2d, SpikingConv1d)):
                self.conv_layers.append(m)

    def forward(self) -> torch.Tensor:
        total_loss = 0.0
        count = 0

        for layer in self.conv_layers:
            if layer.current_probs is None or layer.current_cost_coeff is None:
                continue

            keep_probs = layer.current_probs[:, :, 1]
            cost_coeff = layer.current_cost_coeff

            layer_expected_energy = (keep_probs * cost_coeff.view(1, -1)).sum()
            batch_size = keep_probs.size(0)
            layer_max_energy = cost_coeff.sum() * batch_size + 1e-8

            total_loss += (layer_expected_energy / layer_max_energy)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device, requires_grad=True)

        return self.lambda_energy * (total_loss / count)


class GumbelTemperatureScheduler:
    def __init__(self, model: nn.Module, init_temp: float = 5.0, final_temp: float = 0.1,
                 total_epochs: int = 100):
        self.conv_layers = []
        for m in model.modules():
            if isinstance(m, (SpikingConv2d, SpikingConv1d)):
                self.conv_layers.append(m)
                m.set_temperature(init_temp)

        self.init_temp = init_temp
        self.final_temp = final_temp
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.decay_factor = (self.final_temp / self.init_temp) ** (1.0 / self.total_epochs)

    def step(self):
        self.current_epoch += 1
        new_temp = self.init_temp * (self.decay_factor ** self.current_epoch)
        new_temp = max(new_temp, self.final_temp)

        for layer in self.conv_layers:
            layer.set_temperature(new_temp)

    def get_temp(self):
        if len(self.conv_layers) > 0:
            return self.conv_layers[0].temperature
        return self.init_temp


