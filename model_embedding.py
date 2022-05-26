import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rgcn import RGCN, RGCN_Time


class RGCN_Model(nn.Module):
    def __init__(self, 
        input_size,
        hidden_size,
        output_size,
        num_bases,
        num_rel,
        num_layers,
        dropout,
        device
    ):
        super(RGCN_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.device = device
        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(RGCN(self.input_size, self.hidden_size, self.num_bases, self.num_rel, self.device))
            else:
                if i == self.num_layers - 1:
                    self.layers.append(RGCN(self.hidden_size, self.output_size, self.num_bases, self.num_rel, self.device))
                else:
                    self.layers.append(RGCN(self.hidden_size, self.hidden_size, self.num_bases, self.num_rel, self.device))
        print('Using embedding type: rgcn')

    def forward(self, feature_list, adj_list, alignment_list, train_year):
        embeddings = []
        for t in range(train_year):
            x = feature_list[t].to(self.device)
            y = adj_list[t]
            for i, layer in enumerate(self.layers):
                x = layer(x, y)
                if i != self.num_layers - 1:
                    x = F.dropout(self.relu(x), self.dropout, training=self.training)
                else:
                    x = F.dropout(x, self.dropout, training=self.training)

            embeddings.append(x)

        return  embeddings


# RGCN initialized with previous embeddings
class RGCN_Time_Model(nn.Module):
    def __init__(self, 
        input_size,
        hidden_size,
        output_size,
        num_bases,
        num_rel,
        num_layers,
        dropout,
        device
    ):
        super(RGCN_Time_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.device = device

        for i in range(self.num_layers):
            if i == 0:
                self.layers.append(RGCN_Time(self.input_size, self.hidden_size, self.num_bases, self.num_rel, self.device, self.output_size))
            else:
                if i == self.num_layers - 1:
                    self.layers.append(RGCN_Time(self.hidden_size, self.output_size, self.num_bases, self.num_rel, self.device, self.output_size))
                else:
                    self.layers.append(RGCN_Time(self.hidden_size, self.hidden_size, self.num_bases, self.num_rel, self.device, self.output_size))
        print('Using embedding type: time-rgcn')
    
    def forward(self, feature_list, adj_list, alignment_list, train_year):
        embeddings = []
        x = None
        y = None
        for t in range(train_year):
            if t > 0:
                x = feature_list[t].to(self.device)
                y = adj_list[t]
                align_t = alignment_list[:,t]
                line_nums = (align_t != -1).nonzero(as_tuple=True)[0]
                idx_seqs = alignment_list[line_nums]
                idx_prevs = idx_seqs[:, t-1]

                align_t_prev = alignment_list[:,t-1]
                line_nums_prev = (align_t_prev != -1).nonzero(as_tuple=True)[0]
                idx_seqs_prev = alignment_list[line_nums_prev]
                idx_prev_now = idx_seqs_prev[:, t]

                # emb_prev = [embeddings[t-1][item] for item in idx_prevs if item.item() != -1]
                # emb_prev = torch.stack(emb_prev)

                idx_prevs_flag = idx_prevs >= 0
                idx_prevs_pos = torch.index_select(idx_prevs, 0, torch.squeeze(torch.nonzero(idx_prevs_flag)))
                emb_prev = torch.index_select(embeddings[t-1], 0, idx_prevs_pos)

                adj_prev = adj_list[t-1]
                for i, layer in enumerate(self.layers):
                    x = layer(x, y, emb_prev=emb_prev, adj_prev=adj_prev, idx_prev_now=idx_prev_now)
                    if i != self.num_layers - 1:
                        x = F.dropout(self.relu(x), self.dropout, training=self.training)
                    else:
                        x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = feature_list[t].to(self.device) # num_node * output_size
                y = adj_list[t]
                for i, layer in enumerate(self.layers):
                    x = layer(x, y)
                    if i != self.num_layers - 1:
                        x = F.dropout(self.relu(x), self.dropout, training=self.training)
                    else:
                        x = F.dropout(x, self.dropout, training=self.training)

            embeddings.append(x)

        return  embeddings

