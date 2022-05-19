import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


# Uni-GRU/Bi-GRU, Log/Sigmoid/Tanh
class Traj_Model(nn.Module):
    def __init__(self, imputed_size, device, ts_func_type='log', rnn_type=None):
        super(Traj_Model, self).__init__()
        self.imputed_size = imputed_size
        self.device = device
        self.ts_func_type = ts_func_type
        self.rnn_type = rnn_type
        self.pred_loss = None

        self.mlp_hidden_size = {"base_hidden":20,"base_output":10,"param_hidden_1":20,"param_hidden_2":8,"param_output":1,"rnn":50}

        if self.rnn_type == 'uni':
            self.rnn = nn.GRU(self.imputed_size, self.mlp_hidden_size['rnn'], batch_first=True).to(self.device)
            self.reset_parameters()
        elif self.rnn_type == 'bi':
            self.rnn = nn.GRU(self.imputed_size, self.mlp_hidden_size['rnn'], batch_first=True, bidirectional=True).to(self.device)
            self.fc = nn.Linear(2 * self.mlp_hidden_size['rnn'], self.mlp_hidden_size['rnn'])
            self.relu = nn.ReLU()
            self.reset_parameters()
        else:
            self.rnn_type = None

        if self.rnn_type:
            self.mlp_base = nn.Sequential(
                nn.Linear(self.mlp_hidden_size['rnn'], self.mlp_hidden_size['base_hidden']),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_size['base_hidden'], self.mlp_hidden_size["base_output"])
            )
        else:
            self.mlp_base = nn.Sequential(
                nn.Linear(self.imputed_size, self.mlp_hidden_size['base_hidden']),
                nn.ReLU(),
                nn.Linear(self.mlp_hidden_size['base_hidden'], self.mlp_hidden_size["base_output"])
            )
        self.mlp_base.to(self.device)

        self.mlp_theta1 = nn.Sequential(
            nn.Linear(self.mlp_hidden_size["base_output"], self.mlp_hidden_size["param_hidden_1"]),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size['param_hidden_1'], self.mlp_hidden_size['param_hidden_2']),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size["param_hidden_2"], self.mlp_hidden_size["param_output"]),
            nn.ReLU()
        )
        self.mlp_theta1.to(self.device)

        self.mlp_theta2 = nn.Sequential(
            nn.Linear(self.mlp_hidden_size["base_output"], self.mlp_hidden_size["param_hidden_1"]),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size['param_hidden_1'], self.mlp_hidden_size['param_hidden_2']),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size["param_hidden_2"], self.mlp_hidden_size["param_output"]),
            nn.ReLU()
        )
        self.mlp_theta2.to(self.device)

        self.mlp_theta3 = nn.Sequential(
            nn.Linear(self.mlp_hidden_size["base_output"], self.mlp_hidden_size["param_hidden_1"]),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size['param_hidden_1'], self.mlp_hidden_size['param_hidden_2']),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size["param_hidden_2"], self.mlp_hidden_size["param_output"]),
            nn.ReLU()
        )
        self.mlp_theta3.to(self.device)

        self.mlp_xi = nn.Sequential(
            nn.Linear(self.mlp_hidden_size["base_output"], self.mlp_hidden_size["param_hidden_1"]),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size['param_hidden_1'], self.mlp_hidden_size['param_hidden_2']),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size["param_hidden_2"], self.mlp_hidden_size["param_output"]),
            nn.Softplus()
        )
        self.mlp_xi.to(self.device)

        print("Using TS Function: " + self.ts_func_type +"  Using RNN type: " + str(self.rnn_type))

    def reset_parameters(self):
        for k, v in self.rnn.state_dict().items():
            torch.nn.init.constant_(v, 0)

    # HINTS: https://dl.acm.org/doi/10.1145/3442381.3450107
    def log_survival(self, theta1, theta2, xi, t):
        return theta1*1/(np.sqrt(2*np.pi)*xi*t) * torch.exp(-(np.log(t)-theta2)**2/(2*xi**2))

    def sigmoid_simple(self, theta1, theta2, theta3, t):
        eps = 1e-8
        return theta1 / ((1 + torch.exp(-1 * theta2 * (t - theta3))) + eps)

    # https://en.wikipedia.org/wiki/Generalised_logistic_function
    # may need to try several times for a better initialization point for optimizing
    def logistic_general(self, theta1, theta2, theta3, xi, t):
        return theta1 / torch.pow(1 + xi * torch.exp(-1 * theta2 * (t - theta3)), 1 / xi)

    def forward(self, imputed_embeds, predict_seq, predict_year):
        final_state = None
        citation_pred_list = None

        if self.rnn_type:
            output, final_state = self.rnn(imputed_embeds)
            if self.rnn_type == 'bi':
                final_state = self.fc(self.relu(torch.cat([final_state[-1], final_state[-2]], dim=1))) # batch_size * output_size
        else:
            final_state = imputed_embeds

        base = self.mlp_base(final_state)
        theta1 = self.mlp_theta1(base)
        theta2 = self.mlp_theta2(base)
        theta3 = self.mlp_theta3(base)
        xi = self.mlp_xi(base)

        if predict_year == 0:
            if self.ts_func_type == 'log':
                citation_pred_list = [self.log_survival(theta1, theta2, xi, t) for t in range(1, predict_seq+1)]
            elif self.ts_func_type == 'logistic':
                citation_pred_list = [self.logistic_general(theta1, theta2, theta3, xi, t) for t in range(1, predict_seq+1)]
                # citation_pred_list = [self.sigmoid_simple(theta1, theta2, theta3, t) for t in range(1, predict_seq+1)]
        
            if len(citation_pred_list) > 1:
                self.citation_pred = torch.transpose(torch.squeeze(torch.stack(citation_pred_list)), 0, 1) # batch_size * years
            else:
                tmp = torch.squeeze(torch.stack(citation_pred_list), 1)
                tmp = torch.squeeze(tmp, 2)
                self.citation_pred = torch.transpose(tmp, 0, 1) 

        else:
            citation_pred_list = self.logistic_general(theta1, theta2, theta3, xi, predict_year)
            self.citation_pred = torch.squeeze(citation_pred_list)

        return self.citation_pred



class Traj_Model_Simple(nn.Module):
    def __init__(self, imputed_size, predict_year, device):
        super(Traj_Model_Simple, self).__init__()
        self.imputed_size = imputed_size
        self.predict_year = predict_year
        self.device = device
        self.pred_loss = None
        self.mlp_hidden_size = {"base_hidden":20,"base_output":10,"param_hidden_1":20,"param_hidden_2":8,"param_output":1,"rnn":50}
        self.rnn = nn.GRU(self.imputed_size, self.mlp_hidden_size['rnn'], batch_first=True).to(self.device)
        self.reset_parameters()

        if predict_year == 0:
            self.mlp = nn.Sequential(
                nn.Linear(self.mlp_hidden_size['rnn'], self.predict_year),
                nn.Softplus()
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.mlp_hidden_size['rnn'], 1),
                nn.Softplus()
            )
        self.mlp.to(self.device)
        print("Using TS Function: Linear")

    def reset_parameters(self):
        for k, v in self.rnn.state_dict().items():
            torch.nn.init.constant_(v, 0)

    def forward(self, imputed_embeds, predict_seq, predict_year):
        output, final_state = self.rnn(imputed_embeds)

        self.citation_pred = torch.squeeze(self.mlp(final_state)) # batch_size * years

        return self.citation_pred
