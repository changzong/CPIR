import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TACN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, time_steps, device, kernel_size=2, dropout=0):
        super(TACN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.time_steps = time_steps
        self.device = device
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.tcn = TemporalConv(self.input_size, self.num_channels, kernel_size=self.kernel_size, dropout=self.dropout).to(self.device)
        self.linear = nn.Linear(self.num_channels[-1], self.output_size).to(self.device)
        self.w_key = nn.Linear(self.time_steps, self.time_steps, bias=True).to(self.device)
        self.w_value = nn.Linear(self.time_steps, self.time_steps, bias=False).to(self.device)
        self.init_weights()
        self.post_attention_layer = AttentionLayer(self.output_size, self.device).to(self.device)

    def init_weights(self):
        self.linear.weight.data.uniform_(0.0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = torch.permute(x, (1, 2, 0)) # batch_size * emb_size * time_steps
        y = self.tcn(x) # batch_size * output_size * time_steps
        query = y.to(self.device)
        key = self.w_key(x).to(self.device)
        value = self.w_value(x).to(self.device)
        x = self.post_attention_layer(query, value, key) # batch_size * output_size * time_steps
        return x[:,:,-1]  # batch_size * output_size


class AttentionLayer(nn.Module):
    def __init__(self, scale_value, device):
        super(AttentionLayer, self).__init__()
        self.device = device
        self.scale_value = torch.tensor(scale_value).float().to(self.device)

    def forward(self, query, key, value):
        scores = torch.matmul(query, torch.transpose(key, 1, 2))
        scores = scores / torch.sqrt(self.scale_value)
        dist = torch.nn.functional.softmax(scores)
        output = torch.squeeze(torch.matmul(dist, value), -1)
        return output


class TemporalConv(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConv, self).__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.num_inputs if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, self.kernel_size, dilation=dilation_size, padding=(self.kernel_size-1) * dilation_size, dropout=self.dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.dilation = dilation
        self.conv1 = weight_norm(nn.Conv1d(self.n_inputs, self.n_outputs, self.kernel_size, stride=1, padding=self.padding, dilation=self.dilation))
        self.chomp1 = Chomp1d(self.padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)

        self.conv2 = weight_norm(nn.Conv1d(self.n_outputs, self.n_outputs, self.kernel_size, stride=1, padding=self.padding, dilation=self.dilation))
        self.chomp2 = Chomp1d(self.padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1, self.relu1,
                                 self.conv2, self.chomp2, self.dropout2, self.relu2)

        self.downsample = weight_norm(nn.Conv1d(self.n_inputs, self.n_outputs, 1)) if self.n_inputs != self.n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.01, 0.01)
        self.conv2.weight.data.uniform_(-0.01, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

