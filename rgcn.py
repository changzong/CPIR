
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class RGCN(nn.Module):
    def __init__(self, input_size, output_size, num_bases, num_rel, device, bias=False):
        super(RGCN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.device = device
        self.num_rel = num_rel
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size)).to(self.device)
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases)).to(self.device)
        else:
            self.weight = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size)).to(self.device)
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size)).to(self.device)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj_list): 
        if self.num_bases > 0:
            self.weight = torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
        # shape(r*input_size, output_size)
        weights = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2])  
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            adj = adj_list[i].to(self.device)
            if input is not None:
                supports.append(torch.sparse.mm(adj.float(), input.float()))
            else:
                supports.append(adj)

        tmp = torch.cat(supports, dim=1)
        # shape(#node, output_size)
        output = torch.mm(tmp.float(), weights)

        if self.bias is not None:
            output += self.bias.unsqueeze(0)
        return output


class RGCN_Time(nn.Module):
    def __init__(self, input_size, output_size, num_bases, num_rel, device, prev_size, bias=False):
        super(RGCN_Time, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.device = device
        self.num_rel = num_rel
        self.prev_size = prev_size
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size)).to(self.device)
            self.w_rel = Parameter(torch.FloatTensor(self.num_rel, self.num_bases)).to(self.device)
        else:
            self.weight = Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size)).to(self.device)
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_size)).to(self.device)
        else:
            self.register_parameter("bias", None)
        self.weight_prev = Parameter(torch.FloatTensor(self.num_rel, self.prev_size, self.output_size)).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        nn.init.xavier_uniform_(self.weight_prev.data)

    def forward(self, input, adj_list, emb_prev=None, adj_prev=None, idx_prev_now=None): 
        if self.num_bases > 0:
            self.weight = torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
        # shape(r*input_size, output_size)
        weights = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2]) # num_rel * input_size, output_size
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            adj = adj_list[i].to(self.device)
            if input is not None:
                supports.append(torch.sparse.mm(adj.float(), input.float()))
            else:
                supports.append(adj)

        tmp = torch.cat(supports, dim=1) # num_rel * input_size
        output = torch.mm(tmp.float(), weights) # shape(#node, output_size)

        if emb_prev is not None:
            weights_prev = self.weight_prev.view(self.weight_prev.shape[0] * self.weight_prev.shape[1], self.weight_prev.shape[2])
            supports_prev = []
            for i in range(self.num_rel):
                adj = adj_prev[i].to(self.device)
                supports_prev.append(torch.sparse.mm(adj.float(), emb_prev.float()))

            tmp_prev = torch.cat(supports_prev, dim=1) # num_rel * emb_prev_size
            output_prev = torch.mm(tmp_prev.float(), weights_prev)
            output[idx_prev_now] = output[idx_prev_now] + output_prev

        if self.bias is not None:
            output += self.bias.unsqueeze(0)
        return output
        