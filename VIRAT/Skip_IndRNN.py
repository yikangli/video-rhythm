import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable, Function
from torch.nn import Parameter, ParameterList
import math
import pdb

class BinaryLayer(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.round()
    
    @staticmethod 
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

def check_bounds(weight, min_abs, max_abs):
    if min_abs:
        abs_kernel = torch.abs(weight).clamp_(min=min_abs)
        weight = torch.mul(torch.sign(weight), abs_kernel)
    if max_abs:
        weight = weight.clamp(max=max_abs, min=-max_abs)
    return weight


class SkipIndRNNCell(nn.Module):
    __constants__ = [
        "hidden_max_abs", "hidden_min_abs", "input_size", "hidden_size",
        "nonlinearity", "hidden_init", "recurrent_init",
    ]

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="relu",
                 hidden_min_abs=0, hidden_max_abs=1e3,
                 hidden_init=None, recurrent_init=None,
                 gradient_clip=5.0):
        super(SkipIndRNNCell, self).__init__()
        self.hidden_max_abs = hidden_max_abs
        self.hidden_min_abs = hidden_min_abs
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.hidden_init = hidden_init
        self.recurrent_init = recurrent_init

        self.BL = BinaryLayer.apply

        if self.nonlinearity == "tanh":
            self.activation = torch.tanh
        elif self.nonlinearity == "elu":
            self.activation = F.elu
        elif self.nonlinearity == "relu":
            self.activation = F.relu
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size))
        self.weight_uh = Parameter(torch.Tensor(1, hidden_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_uh = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_uh', None)

        if gradient_clip:
            if isinstance(gradient_clip, tuple):
                min_g, max_g = gradient_clip
            else:
                max_g = gradient_clip
                min_g = -max_g
            self.weight_ih.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_hh.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            self.weight_uh.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
            if bias:
                self.bias_ih.register_hook(lambda x: x.clamp(min=min_g, max=max_g))
                self.bias_uh.register_hook(lambda x: x.clamp(min=min_g, max=max_g))

        self.reset_parameters()

    def check_bounds(self):
        self.weight_hh.data = check_bounds(self.weight_hh.data, self.hidden_min_abs, self.hidden_max_abs)

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if "bias" in name:
                weight.data.zero_()
            elif "weight_hh" in name:
                if self.recurrent_init is None:
                    nn.init.constant_(weight, 1)
                else:
                    self.recurrent_init(weight)
            elif "weight_ih" in name:
                if self.hidden_init is None:
                    nn.init.normal_(weight, 0, 0.01)
                else:
                    self.hidden_init(weight)
            elif "weight_uh" in name:
                if self.hidden_init is None:
                    #nn.init.normal_(weight, 0, 0.01)
                    nn.init.xavier_normal_(weight, gain=np.sqrt(2))
                else:
                    self.hidden_init(weight)
            else:
                weight.data.normal_(0, 0.01)
                # weight.data.uniform_(-stdv, stdv)
        self.check_bounds()

    def forward(self, input, h):
        #pdb.set_trace()
        hx, update_prob_prev, cum_update_prob_prev = h
        new_hx = self.activation(F.linear(input, self.weight_ih, self.bias_ih) + torch.mul(self.weight_hh, hx))

        # Compute value for the update prob
        new_update_prob_tilde = torch.sigmoid(torch.mul(new_hx, self.weight_uh) + self.bias_uh)
        #new_update_prob_tilde = torch.sigmoid(F.linear(new_hx, self.weight_uh, self.bias_uh))

        # Compute value for the update gate
        cum_update_prob = cum_update_prob_prev + torch.min(update_prob_prev, 1. - cum_update_prob_prev)
        # round
        #bn = self.BinaryLayer()
        update_gate = self.BL(cum_update_prob)
        # Apply update gate
        new_h = update_gate * new_hx + (1. - update_gate) * hx

        new_update_prob = update_gate * new_update_prob_tilde + (1. - update_gate) * update_prob_prev
        new_cum_update_prob = update_gate * 0. + (1. - update_gate) * cum_update_prob
        #new_states = torch.cat((new_h, new_update_prob, new_cum_update_prob),dim=1)
        new_states = (new_h, new_update_prob, new_cum_update_prob)
        #new_output = torch.cat((new_h, update_gate),dim=1)
        new_output = (new_h, update_gate)

        return new_output, new_states


class SkipIndRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer=1, batch_norm=False,
                 batch_first=False, bidirectional=False,
                 hidden_inits=None, recurrent_inits=None,
                 **kwargs):
        super(SkipIndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_norm = batch_norm
        self.n_layer = n_layer
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.num_directions = num_directions = 2 if self.bidirectional else 1

        if batch_first:
            self.time_index = 1
            self.batch_index = 0
        else:
            self.time_index = 0
            self.batch_index = 1

        cells = []
        cells_bi = []
        for i in range(n_layer):
            if recurrent_inits is not None:
                kwargs["recurrent_init"] = recurrent_inits[i]
            if hidden_inits is not None:
                kwargs["hidden_init"] = hidden_inits[i]
            in_size = input_size if i == 0 else hidden_size * num_directions
            cells.append(SkipIndRNNCell(in_size, hidden_size, **kwargs))
            cells_bi.append(SkipIndRNNCell(in_size, hidden_size, **kwargs))
        self.cells = nn.ModuleList(cells)
        self.cells_bi = nn.ModuleList(cells_bi)

        if batch_norm:
            bns = []
            for i in range(n_layer):
                bns.append(nn.BatchNorm1d(hidden_size * num_directions))
            self.bns = nn.ModuleList(bns)
        
        # self.initial_states = []
        # for i in np.arange(n_layer):
        #     initial_h = torch.randn(hidden_size*num_directions, requires_grad=False)
        #     if i == self.n_layer - 1: # last layer
        #         initial_update_prob = torch.ones((batch_size, 1), requires_grad=False)
        #         initial_cum_update_prob = torch.zeros((batch_size, 1), requires_grad=False)
        #     else:
        #         initial_update_prob = torch.zeros((1,hidden_size), requires_grad=False)
        #         initial_cum_update_prob = torch.zeros((1,hidden_size),  requires_grad=False)
        #     self.initial_states.append((initial_h, initial_update_prob, initial_cum_update_prob))
        #h0 = torch.zeros(hidden_size * num_directions, requires_grad=False)
        #self.register_buffer('h0', h0)
        #self.register_buffer('initial_states',initial_states)

    def forward(self, x, hidden=torch.tensor(float("nan"))):
        batch_norm = self.batch_norm
        time_index = self.time_index
        batch_index = self.batch_index
        num_directions = self.num_directions
        hiddens = []
        i = 0
        batch_size = x.shape[1]

        self.initial_states = []
        for i in np.arange(self.n_layer):
            initial_h = torch.randn(self.hidden_size*num_directions, requires_grad=False)
            if i == self.n_layer - 1: # last layer
                initial_update_prob = torch.ones((batch_size, self.hidden_size), requires_grad=False)
                initial_cum_update_prob = torch.zeros((batch_size, self.hidden_size), requires_grad=False)
            else:
                # initial_update_prob = torch.zeros((1,hidden_size), requires_grad=False)
                # initial_cum_update_prob = torch.zeros((1,hidden_size),  requires_grad=False)
                initial_update_prob = torch.ones((batch_size, self.hidden_size), requires_grad=False)
                initial_cum_update_prob = torch.zeros((batch_size, self.hidden_size), requires_grad=False)
            self.initial_states.append((initial_h, initial_update_prob, initial_cum_update_prob))

        if x.is_cuda:
            if len(self.cells) == 1:
                self.initial_states = tuple([y.cuda() for y in self.initial_states])
            else:
                self.initial_states = [tuple([j.cuda() if j is not None else None for j in i]) for i in self.initial_states]
        #pdb.set_trace()
        #for cell in self.cells:
        x_n = [x]
        x_updates = []
        for i in range(len(self.cells)):
            #hx = self.h0.unsqueeze(0).expand(x.size(batch_index),self.hidden_size * num_directions).contiguous()
            cell = self.cells[i]
            hidden_state = self.initial_states[i]
            hx_ = hidden_state[0].unsqueeze(0).expand(x.size(batch_index),self.hidden_size * num_directions).contiguous()
            hx = hx_[:, : self.hidden_size * 1]
            hx_bi = hx_[:, self.hidden_size: self.hidden_size * 2]
            hx_cell = (hx, hidden_state[1], hidden_state[2])
            hx_cell_bi = (hx_bi, hidden_state[1], hidden_state[2])

            cell.weight_hh.data = check_bounds(
                cell.weight_hh.data, cell.hidden_min_abs, cell.hidden_max_abs
            )
            outputs = []
            lst_update_gates = []
            x_T = torch.unbind(x_n[i], time_index)
            time_frame = len(x_T)
            for t in range(time_frame):
                output_cell,hx_cell = cell(x_T[t], hx_cell)
                new_h, update_gate = output_cell
                outputs.append(new_h)
                lst_update_gates.append(update_gate)
            #pdb.set_trace()
            x_cell = torch.stack(outputs, time_index)
            update_gates_cell = torch.stack(lst_update_gates, time_index)
            if self.bidirectional:
                outputs_bi = []
                lst_update_gates_bi = []
                for t in range(time_frame - 1, -1, -1):
                    output_cell_bi, hx_cell_bi = self.cells_bi[i](x_T[t], hx_cell_bi)
                    new_h_bi, update_gate_bi = output_cell_bi
                    outputs_bi.append(new_h_bi)
                    lst_update_gates_bi.append(update_gate_bi)
                x_cell_bi = torch.stack(outputs_bi[::-1], time_index)
                x_cell = torch.cat([x_cell, x_cell_bi], 2)
                update_gates_cell_bi = torch.stack(lst_update_gates_bi[::-1],time_index)
                update_gates_cell = torch.cat([update_gates_cell, update_gates_cell_bi], 2)
            #pdb.set_trace()
            x_n.append(x_cell)
            x_updates.append(update_gates_cell)
            hiddens.append(hx_cell[0])
            #i += 1
        #pdb.set_trace()
        x = torch.stack(x_n[1:], 0)
        x_up = torch.stack(x_updates, 0)
        #pdb.set_trace()
        if batch_norm:
            if self.batch_first:
                x = self.bns[i](
                    x.permute(batch_index, 2, time_index).contiguous()).permute(0, 2, 1)
            else:
                x = self.bns[i](
                    x.permute(batch_index, 2, time_index).contiguous()).permute(2, 0, 1)
        return x.squeeze(), x_up.squeeze(), torch.cat(hiddens, -1)