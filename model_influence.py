import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from tacn import TACN

import pdb

# normal influence (HINTS)
class Static_Influence_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, num_rel, device):
        super(Static_Influence_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.num_rel = num_rel
        self.device = device

        self.weights = [Parameter(torch.FloatTensor(self.graph_emb_size, self.influence_emb_size)).to(self.device) for i in range(self.num_rel)]
        self.reset_parameters()

        print('Using influence type: static')

    def reset_parameters(self):
        self.initial_weights = []
        for weight in self.weights:
            nn.init.xavier_uniform_(weight.data)
            self.initial_weights.append(weight)


    def forward(self, embeddings, train_year, index_list, input_ids, alignment_list, neighbors):
        influence_embeddings = []
        for i in range(train_year):
            output_mean = []
            for j, weight in enumerate(self.initial_weights):
                idx_rel_list = [index_list[i][j][idx.item()] for idx in input_ids] # input_size * rel_size(arbitrary)
                tmp = []
                num_idx_rel_list = []
                for idx_rel in idx_rel_list:
                    output_rel = torch.zeros([1, self.graph_emb_size]).to(self.device)
                    idx_rel_tensor = torch.tensor(idx_rel).to(self.device)
                    if idx_rel:
                        output_rel = torch.add(output_rel, torch.sum(torch.index_select(embeddings[i], 0, idx_rel_tensor), 0)) # 1 * embed_size
                        num_idx_rel_list.append(len(idx_rel))
                    else:
                        num_idx_rel_list.append(1)
                    tmp.append(output_rel)
                num_idx_rel = torch.tensor(num_idx_rel_list).to(self.device) # input_size * 1
                num_idx_rel = torch.repeat_interleave(num_idx_rel, self.graph_emb_size, dim=0).reshape(len(num_idx_rel_list), self.graph_emb_size).to(self.device) # input_size * embed_size
                output_sum_rel = torch.cat(tmp).to(self.device) # input_size * embed_size
                output_mean_rel = torch.div(output_sum_rel, num_idx_rel) # input_size * embed_size
                output_mean_rel = torch.matmul(output_mean_rel, weight) # input_size * influence_size
                output_mean.append(output_mean_rel) # rel_size * input_size * influence_size

            influence_embeddings.append(torch.sum(torch.stack(output_mean), 0)) # year_size * input_size * influence_size

        influence_embeddings = torch.stack(influence_embeddings)
        influence_embeddings = torch.reshape(influence_embeddings, (-1, train_year, self.influence_emb_size)) # input_size * year_size * influence_size

        return influence_embeddings


# normal influence (HDGNN approximate)
class HDGNN_Influence_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, num_rel, batch_size, device):
        super(HDGNN_Influence_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.num_rel = num_rel
        self.device = device
        self.hidden_size = 128
        self.output_size = 64
        self.num_heads = 2

        self.mlp = nn.Sequential(
            nn.Linear(self.graph_emb_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ).to(self.device)
        self.gru = nn.GRU(32, self.output_size, batch_first=True, bidirectional=True).to(self.device)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads).to(self.device)
        self.query = Parameter(torch.FloatTensor(5, batch_size, self.influence_emb_size)).to(self.device)

        self.reset_parameters()

        print('Using influence type: hdgnn')

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.query.data)
        for name, param in self.gru.named_parameters():
            nn.init.uniform_(param, -0.1, 0.1)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, embeddings, train_year, index_list, input_ids, alignment_list, neighbors):
        influence_embeddings = []
        for i in range(train_year):
            output_feature = []
            for j in range(self.num_rel):
                idx_rel_list = [index_list[i][j][idx.item()] for idx in input_ids] # batch_size * rel_size(arbitrary)
                tmp = []
                for idx_rel in idx_rel_list:
                    idx_rel_tensor = torch.tensor(idx_rel).to(self.device)
                    seq_output_rel = []
                    if idx_rel:
                        hidden = self.mlp(torch.index_select(embeddings[i], 0, idx_rel_tensor))
                        output, final_state = self.gru(torch.unsqueeze(hidden, 0))
                        tmp.append(torch.cat([final_state[-1], final_state[-2]], dim=1))
                    else:
                        tmp.append(torch.zeros([1, self.hidden_size]).to(self.device))
                output_feature.append(torch.squeeze(torch.stack(tmp), 1))
            
            # output_mean = torch.mean(torch.stack(output_feature), 0)
            tmp_emb = embeddings[i]
            tmp_emb[0] = torch.zeros(1, self.graph_emb_size)
            idxs = alignment_list[input_ids][:, i]
            idxs = torch.where(idxs>=0, idxs, 0)
            self_features = torch.unsqueeze(torch.index_select(embeddings[i], 0, idxs), 0)
            value = torch.cat([self_features, torch.stack(output_feature)], 0)
            key = value
            attn_output, output_weights = self.multihead_attn(self.query, key, value)
            influence_embeddings.append(torch.mean(attn_output, 0))
        influence_embeddings = torch.stack(influence_embeddings)
        influence_embeddings = torch.reshape(influence_embeddings, (-1, train_year, self.influence_emb_size))
        return influence_embeddings


# citation relation can be produced by 3 types: novelty relevant, inventive step relevant, general technical background relevant
# implict citation influence
class Static_Plus_Influence_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, rel_types, device):

        super(Static_Plus_Influence_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.rel_types = rel_types
        self.num_rel = len(rel_types)
        self.cite_idx = self.rel_types.index('citedBy')
        self.device = device

        self.weights = [Parameter(torch.FloatTensor(self.graph_emb_size, self.influence_emb_size)).to(self.device) for i in range(self.num_rel-1)]
        self.weights_cite = [Parameter(torch.FloatTensor(self.graph_emb_size, self.influence_emb_size)).to(self.device) for i in range(3)]

        self.reset_parameters()

    def reset_parameters(self):
        self.initial_weights = []
        self.initial_weights_cite = []
        for weight in self.weights:
            nn.init.xavier_uniform_(weight.data)
            self.initial_weights.append(weight)
        for weight in self.weights_cite:
            nn.init.xavier_uniform_(weight.data)
            self.initial_weights_cite.append(weight)

    def forward(self, embeddings, train_year, index_list, input_ids):
        influence_embeddings = []
        for i in range(train_year):
            weighted_output = []
            for rel_idx, rel_weight in enumerate(self.initial_weights):
                rel_idx_batch = [index_list[i][rel_idx][idx.item()] for idx in input_ids] # input_size * rel_size(arbitrary)
                rel_sum_batch = []
                rel_idx_num_batch = []
                for rel_idx_list in rel_idx_batch:
                    rel_sum_output = torch.zeros([1, self.graph_emb_size]).to(self.device)
                    rel_idx_list_tensor = torch.tensor(rel_idx_list).to(self.device)
                    if rel_idx_list:
                        rel_sum_output = torch.add(rel_sum_output, torch.sum(torch.index_select(embeddings[i], 0, rel_idx_list_tensor), 0)) # 1 * embed_size
                        rel_idx_num_batch.append(len(rel_idx_list))
                    else:
                        rel_idx_num_batch.append(1)
                    rel_sum_batch.append(rel_sum_output)
                rel_idx_num_batch_tensor = torch.tensor(rel_idx_num_batch).to(self.device) # input_size * 1
                rel_idx_num_batch_tensor = torch.repeat_interleave(rel_idx_num_batch_tensor, self.graph_emb_size, dim=0).reshape(len(rel_idx_num_batch), self.graph_emb_size).to(self.device) # input_size * embed_size
                rel_sum_batch_tensor = torch.cat(rel_sum_batch).to(self.device) # input_size * embed_size
                rel_mean_batch_tensor = torch.div(rel_sum_batch_tensor, rel_idx_num_batch_tensor) # input_size * embed_size
                if rel_idx == self.cite_idx:
                    for k, cite_weight in enumerate(self.initial_weights_cite):
                        rel_weighted_batch_tensor = torch.matmul(rel_mean_batch_tensor, cite_weight) # input_size * influence_size
                        weighted_output.append(rel_weighted_batch_tensor) # rel_size * input_size * influence_size
                else:
                    rel_weighted_batch_tensor = torch.matmul(rel_mean_batch_tensor, rel_weight) # input_size * influence_size
                    weighted_output.append(rel_weighted_batch_tensor) # rel_size * input_size * influence_size

            influence_embeddings.append(torch.sum(torch.stack(weighted_output), 0)) # year_size * input_size * influence_size

        influence_embeddings = torch.stack(influence_embeddings)
        influence_embeddings = torch.reshape(influence_embeddings, (-1, train_year, self.influence_emb_size)) # input_size * year_size * influence_size

        return influence_embeddings


# Bi-RNN for each neighbors of input, and then influence by weight
class Dynamic_Influence_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, num_rel, device):
        super(Dynamic_Influence_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.num_rel = num_rel
        self.device = device
        self.hidden_size = 128
        self.output_size = 64

        self.rnns = []
        self.fcs = []
        self.weights = []

        for i in range(self.num_rel):
            rnn = nn.LSTM(self.graph_emb_size, self.hidden_size, bidirectional=True).to(self.device)
            for name, param in rnn.named_parameters():
                nn.init.uniform_(param, -0.1, 0.1)
            # for k, v in rnn.state_dict().items():
            #    nn.init.constant_(v, 0)
            self.rnns.append(rnn)

            layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * self.hidden_size, self.output_size)
            ).to(self.device)
            self.fcs.append(layer)

            weight1 = Parameter(torch.FloatTensor(self.output_size, self.influence_emb_size)).to(self.device)
            nn.init.xavier_uniform_(weight1.data)
            weight2 = Parameter(torch.FloatTensor(self.output_size, self.influence_emb_size)).to(self.device)
            nn.init.xavier_uniform_(weight2.data)
            weight_rel = Parameter(torch.FloatTensor(self.influence_emb_size, self.influence_emb_size)).to(self.device)
            nn.init.xavier_uniform_(weight_rel.data)
            self.weights.append([weight1, weight2, weight_rel])

        print('Using influence type: dynamic') 

    def forward(self, embeddings, train_year, index_list, input_ids, alignment_list, neighbors):
        influence_embeddings = []
        for input_id in input_ids:
            influence_output = []
            neighbors_input = neighbors[input_id.item()]
            for rel_type_idx, rel_weight in enumerate(self.weights):
                rel_idx_seq = [item[rel_type_idx][input_id.item()] for item in index_list] # year * rel_size(arbitrary)
                rel_neighbors_idx = torch.tensor(neighbors_input[rel_type_idx]).type(torch.int64).to(self.device)
                if rel_neighbors_idx.shape[0] == 0:
                    continue
                neighbor_seq = alignment_list[rel_neighbors_idx] # neighbor_num_rel(batch) * year
                emb_neighbor_seq = []
                for t in range(train_year):
                    neighbor_t = neighbor_seq[:, t]
                    emb_neighbor_t_1 = torch.index_select(embeddings[t], 0, neighbor_t[neighbor_t >= 0])
                    emb_neighbor_t_2 = torch.zeros([list(neighbor_t[neighbor_t < 0].size())[0], self.graph_emb_size]).to(self.device)
                    emb_neighbor_seq.append(torch.cat((emb_neighbor_t_1, emb_neighbor_t_2), 0)) # year * neighbor_num_rel * graph_emb_size
                emb_neighbor_seq_norm = nn.functional.normalize(torch.stack(emb_neighbor_seq))
                output, final_state = self.rnns[rel_type_idx](emb_neighbor_seq_norm)
                final_state = final_state[0]
                output_state_neighbor = self.fcs[rel_type_idx](torch.cat([final_state[-1], final_state[-2]], dim=1)) # neighbor_num_rel * output_size
                rel_influence1 = torch.matmul(output_state_neighbor, rel_weight[0])
                rel_influence2 = torch.matmul(output_state_neighbor, rel_weight[1])
                rel_influence = rel_influence1 + rel_influence2
                influence_output.append(torch.sum(torch.matmul(rel_influence, rel_weight[2]), 0)) # rel_num * influence_size

                # influence_output.append(torch.sum(torch.matmul(output_state_neighbor, rel_weight), 0)) # rel_num * influence_size
            influence_embeddings.append(torch.sum(torch.stack(influence_output), 0)) # batch_size * influence_size
        return torch.stack(influence_embeddings)


class Dynamic_Influence_Coarse_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, num_rel, device):
        super(Dynamic_Influence_Coarse_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.num_rel = num_rel
        self.device = device
        self.hidden_size = 128
        self.output_size = 64

        self.rnns = []
        self.fcs = []
        self.weights = []

        for i in range(self.num_rel):
            rnn = nn.LSTM(self.graph_emb_size, self.hidden_size, bidirectional=True).to(self.device)
            for name, param in rnn.named_parameters():
                nn.init.uniform_(param, -0.1, 0.1)
            # for k, v in rnn.state_dict().items():
            #    nn.init.constant_(v, 0)
            self.rnns.append(rnn)

            layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * self.hidden_size, self.output_size)
            ).to(self.device)
            self.fcs.append(layer)

            weight = Parameter(torch.FloatTensor(self.output_size, self.influence_emb_size)).to(self.device)
            nn.init.xavier_uniform_(weight.data)
            self.weights.append(weight)

        print('Using influence type: dynamic-coarse') 

    def forward(self, embeddings, train_year, index_list, input_ids, alignment_list, neighbors):
        influence_embeddings = []
        for input_id in input_ids:
            influence_output = []
            neighbors_input = neighbors[input_id.item()]
            for rel_type_idx, rel_weight in enumerate(self.weights):
                rel_idx_seq = [item[rel_type_idx][input_id.item()] for item in index_list] # year * rel_size(arbitrary)
                rel_neighbors_idx = torch.tensor(neighbors_input[rel_type_idx]).type(torch.int64).to(self.device)
                if rel_neighbors_idx.shape[0] == 0:
                    continue
                neighbor_seq = alignment_list[rel_neighbors_idx] # neighbor_num_rel(batch) * year
                emb_neighbor_seq = []
                for t in range(train_year):
                    neighbor_t = neighbor_seq[:, t]
                    emb_neighbor_t_1 = torch.index_select(embeddings[t], 0, neighbor_t[neighbor_t >= 0])
                    emb_neighbor_t_2 = torch.zeros([list(neighbor_t[neighbor_t < 0].size())[0], self.graph_emb_size]).to(self.device)
                    emb_neighbor_seq.append(torch.cat((emb_neighbor_t_1, emb_neighbor_t_2), 0)) # year * neighbor_num_rel * graph_emb_size
                emb_neighbor_seq_norm = nn.functional.normalize(torch.stack(emb_neighbor_seq))
                output, final_state = self.rnns[rel_type_idx](emb_neighbor_seq_norm)
                final_state = final_state[0]
                output_state_neighbor = self.fcs[rel_type_idx](torch.cat([final_state[-1], final_state[-2]], dim=1)) # neighbor_num_rel * output_size
                influence_output.append(torch.sum(torch.matmul(output_state_neighbor, rel_weight), 0)) # rel_num * influence_size
            influence_embeddings.append(torch.sum(torch.stack(influence_output), 0)) # batch_size * influence_size
        return torch.stack(influence_embeddings)


# TACN for each neighbors of input, and then influence by weight
class TACN_Influence_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, num_rel, device):
        super(TACN_Influence_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.num_rel = num_rel
        self.device = device
        self.hidden_size = 128
        self.output_size = 64
        self.num_channels = [self.output_size] * 8 # num_level * num_channel
        self.time_steps = 10

        self.tacns = []
        self.weights = []

        for i in range(self.num_rel):
            tacn = TACN(self.graph_emb_size, self.hidden_size, self.num_channels, self.time_steps, self.device, kernel_size=2, dropout=0).to(self.device)
            self.tacns.append(tacn)

            weight = Parameter(torch.FloatTensor(self.output_size, self.influence_emb_size)).to(self.device)
            nn.init.xavier_uniform_(weight.data)
            self.weights.append(weight)

        print('Using influence type: tacn') 

    def forward(self, embeddings, train_year, index_list, input_ids, alignment_list, neighbors):
        alignment_list = alignment_list.to(self.device)
        influence_embeddings = []
        for input_id in input_ids:
            influence_output = []
            neighbors_input = neighbors[input_id.item()]
            for rel_type_idx, rel_weight in enumerate(self.weights):
                rel_idx_seq = [item[rel_type_idx][input_id.item()] for item in index_list] # year * rel_size(arbitrary)
                rel_neighbors_idx = torch.tensor(neighbors_input[rel_type_idx]).type(torch.int64).to(self.device)
                if rel_neighbors_idx.shape[0] == 0:
                    continue
                neighbor_seq = alignment_list[rel_neighbors_idx] # neighbor_num_rel(batch) * year
                emb_neighbor_seq = []
                for t in range(train_year):
                    neighbor_t = neighbor_seq[:, t]
                    emb_neighbor_t_1 = torch.index_select(embeddings[t], 0, neighbor_t[neighbor_t >= 0])
                    emb_neighbor_t_2 = torch.zeros([list(neighbor_t[neighbor_t < 0].size())[0], self.graph_emb_size]).to(self.device)
                    emb_neighbor_seq.append(torch.cat((emb_neighbor_t_1, emb_neighbor_t_2), 0)) # year * neighbor_num_rel * graph_emb_size
                output_state_neighbor = self.tacns[rel_type_idx](torch.stack(emb_neighbor_seq)) # neighbor_num_rel * output_size
                influence_output.append(torch.sum(torch.matmul(output_state_neighbor, rel_weight), 0)) # rel_num * influence_size
            influence_embeddings.append(torch.sum(torch.stack(influence_output), 0)) # batch_size * influence_size
        return torch.stack(influence_embeddings)


class No_Influence_Model(nn.Module):
    def __init__(self, influence_emb_size, graph_emb_size, num_rel, device):
        super(No_Influence_Model, self).__init__()
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.device = device
        print('Using influence type: no') 

    def forward(self, embeddings, train_year, index_list, input_ids, alignment_list, neighbors):
        influence_embeddings = []
        for t in range(train_year):
            align_t = alignment_list[:,t]
            align_input_t = align_t[input_ids]
            line_nums_exist = (align_input_t != -1).nonzero(as_tuple=True)[0]
            line_nums_non_exist = (align_input_t == -1).nonzero(as_tuple=True)[0]

            output = torch.zeros([list(input_ids.size())[0], self.graph_emb_size]).to(self.device)
            output[line_nums_exist] = torch.index_select(embeddings[t], 0, line_nums_exist)
            influence_embeddings.append(output)
        tmp = torch.stack(influence_embeddings)
        return tmp.permute(1,0,2)
