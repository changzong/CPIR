import torch
import torch.nn as nn
from collections import OrderedDict
from model_embedding import RGCN_Model, RGCN_Time_Model
from model_influence import Static_Influence_Model, Static_Plus_Influence_Model, Dynamic_Influence_Model, Dynamic_Influence_Coarse_Model, TACN_Influence_Model, No_Influence_Model
from model_trajectory import Traj_Model, Traj_Model_Simple

# Your future trajectory depends on your history status
class Model(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rel_types,
        num_layer,
        dropout,
        influence_emb_size,
        graph_emb_size,
        batch_size,
        time_steps_history,
        time_steps_predict,
        predict_year,
        device,
        conf
    ):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_bases = 2
        self.rel_types = rel_types
        self.num_rel = len(rel_types)
        self.num_layers = 2
        self.dropout = dropout
        self.influence_emb_size = influence_emb_size
        self.graph_emb_size = graph_emb_size
        self.batch_size = batch_size
        self.time_steps_history = time_steps_history
        self.time_steps_predict = time_steps_predict
        self.predict_year = predict_year
        self.device = device
        self.conf = conf
        self.pred_loss = None
        self.citation_pred = None

        if self.conf['emb_mode'] == 'rgcn':
            rgcn_model = RGCN_Model(self.input_size, self.hidden_size, self.graph_emb_size, self.num_bases, self.num_rel, self.num_layers, self.dropout, self.device)
        elif self.conf['emb_mode'] == 'rgcn-hist':
            rgcn_model = RGCN_Time_Model(self.input_size, self.hidden_size, self.graph_emb_size, self.num_bases, self.num_rel, self.num_layers, self.dropout, self.device)

        if self.conf['impute_mode'] == 'static':
            imputed_model = Static_Influence_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)
        elif self.conf['impute_mode'] == 'dynamic':
            imputed_model = Dynamic_Influence_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)
        elif self.conf['impute_mode'] == 'dynamic-co':
            imputed_model = Dynamic_Influence_Coarse_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)
        elif self.conf['impute_mode'] == 'static-plus':
            imputed_model = Static_Plus_Influence_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)
        elif self.conf['impute_mode'] == 'dynamic-plus':
            imputed_model = Dynamic_Plus_Influence_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)
        elif self.conf['impute_mode'] == 'tacn':
            imputed_model = TACN_Influence_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)
        elif self.conf['impute_mode'] == 'no':
            imputed_model = No_Influence_Model(self.influence_emb_size, self.graph_emb_size, self.num_rel, self.device)

        if self.conf['ts_mode'] == 'log':
            ts_model = Traj_Model(self.influence_emb_size, self.device, ts_func_type='log')
        elif self.conf['ts_mode'] == 'logistic':
            ts_model = Traj_Model(self.influence_emb_size, self.device, ts_func_type='logistic')
        elif self.conf['ts_mode'] == 'hist-log':
            ts_model = Traj_Model(self.influence_emb_size, self.device, ts_func_type='log', rnn_type='uni')
        elif self.conf['ts_mode'] == 'hist-logistic':
            ts_model = Traj_Model(self.influence_emb_size, self.device, ts_func_type='logistic', rnn_type='uni')
        elif self.conf['ts_mode'] == 'bi-hist-log':
            ts_model = Traj_Model(self.influence_emb_size, self.device, ts_func_type='log', rnn_type='bi')
        elif self.conf['ts_mode'] == 'bi-hist-logistic':
            ts_model = Traj_Model(self.influence_emb_size, self.device, ts_func_type='logistic', rnn_type='bi')
        elif self.conf['ts_mode'] == 'simple':
            ts_model = Traj_Model_Simple(self.influence_emb_size, self.predict_year, self.device)

        self.module_list = nn.ModuleList([rgcn_model, imputed_model, ts_model])

    def MALE_loss(self, pred, output):
        pred = pred + 1
        output = output.float() +1
        pred_correct = torch.FloatTensor(pred.size()).type_as(pred)
        mask = torch.abs(pred) < 0.1
        pred_correct[mask] = 0.1
        mask = torch.abs(pred) >= 0.1
        pred_correct[mask] = pred[mask]
        loss = torch.mean(torch.abs(torch.log(pred_correct) - torch.log(output)))
        return loss

    # Managing infinite values from divided by zero or logarithm zero cases
    def MAPE_loss(self, pred, output):
        pred = pred + 1
        output = output.float() +1
        pred_correct = torch.FloatTensor(pred.size()).type_as(pred)
        mask = torch.abs(pred) < 0.1
        pred_correct[mask] = 0.1
        mask = torch.abs(pred) >= 0.1
        pred_correct[mask] = pred[mask]
        loss = torch.mean(torch.abs((torch.log(pred_correct) - torch.log(output) / output)))
        return loss

    def RMLSE_loss(self, pred, output):
        loss_fn = nn.MSELoss()
        pred = pred + 1
        output = output.float() +1
        pred_correct = torch.FloatTensor(pred.size()).type_as(pred)
        mask = torch.abs(pred) < 0.1
        pred_correct[mask] = 0.1
        mask = torch.abs(pred) >= 0.1
        pred_correct[mask] = pred[mask]
        loss = loss_fn(torch.log(pred_correct), torch.log(output))
        loss = torch.sqrt(loss)
        return loss

    def forward(self, adj_list, feature_list, index_list, alignment_list, output_seq, intput_idx, neighbors, flag='train'):
        alignment_list = alignment_list.to(self.device)
        for i, module in enumerate(self.module_list):
            if i == 0:
                graph_embeddings = module(feature_list, adj_list, alignment_list, self.time_steps_history)
            elif i == 1:
                influence_embeddings = module(graph_embeddings, self.time_steps_history, index_list, intput_idx, alignment_list, neighbors)
            elif i == 2:
                citation_pred = module(influence_embeddings, self.time_steps_predict, self.predict_year)
            else:
                continue

        loss = None
        if flag == 'train':
            if self.conf['loss_func'] == 'RMLSE':
                loss = self.RMLSE_loss(citation_pred, output_seq)
            elif self.conf['loss_func'] == 'MALE':
                loss = self.MALE_loss(citation_pred, output_seq)
            elif self.conf['loss_func'] == 'MAPE':
                loss = self.MAPE_loss(citation_pred, output_seq)
        else:
            if self.conf['loss_func'] == 'RMLSE':
                loss = self.RMLSE_loss(citation_pred, output_seq)
            elif self.conf['loss_func'] == 'MALE':
                loss = self.MALE_loss(citation_pred, output_seq)
            elif self.conf['loss_func'] == 'MAPE':
                loss = self.MAPE_loss(citation_pred, output_seq)
        
        return loss, citation_pred
