import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse

from model import Model
from data_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="../data/AIPatent", help='dataset dir')
parser.add_argument('--start_year', type=int, default=2012, help='start year')
parser.add_argument('--time_steps_history', type=int, default=10, help='time_steps')
parser.add_argument('--time_steps_predict', type=int, default=10, help='time_steps')
parser.add_argument('--predict_year', type=int, default=0, help='the nth year')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='valid batch size')
parser.add_argument('--num_layer', type=int, default=2, help='GCN embedding layers')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--influence_emb_size', type=int, default=128, help='imputed embedding vector dimension')
parser.add_argument('--graph_emb_size', type=int, default=128, help='embedding vector dimension')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--use_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--emb_mode', type=str, default='rgcn', help='network embedding method: rgcn/rgcn_hist')
parser.add_argument('--impute_mode', type=str, default='static', help='imputation method: static/dynamic')
parser.add_argument('--ts_mode', type=str, default='simple', help='time-series method: simple/log/logistic')
parser.add_argument('--loss_func', type=str, default='RMLSE', help='loss function type')
args = parser.parse_args()

device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

class PatentCite(Dataset):
    def __init__(self, idx, seq, year):
        self.idx = idx
        self.seq = seq
        self.year = year
  
    def __len__(self):
        return len(self.idx)
  
    def __getitem__(self, index):
        if self.year == 0:
            # seq_list = [list(item.values())[0] for item in self.seq[index]]
            return self.idx[index], self.seq[index]
        else:
            return self.idx[index], self.seq[index][self.year-1]




adj_list, feature_list, index_list, alignment_list, labels, rel_types, index_name, name_index = get_graph_label(args.data_dir, args.start_year, args.time_steps_history, args.time_steps_predict)
train_neighbors = get_neighbors(args.data_dir, labels['train_ids'], args.start_year, rel_types)
valid_neighbors = get_neighbors(args.data_dir, labels['test_ids'], args.start_year, rel_types)

num_rel = len(rel_types)

model_assemble_conf = {
    'emb_mode': args.emb_mode,
    'impute_mode': args.impute_mode,
    'ts_mode': args.ts_mode,
    'loss_func': args.loss_func
}

train_dataset = PatentCite(labels['train_ids'], labels['train_seqs'], args.predict_year)
valid_dataset = PatentCite(labels['test_ids'], labels['test_seqs'], args.predict_year)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, drop_last=True)

input_size = feature_list[0].shape[1]

model = Model(input_size, args.hidden_size, rel_types, args.num_layer, args.dropout, args.influence_emb_size, args.graph_emb_size, args.batch_size, args.time_steps_history, args.time_steps_predict, args.predict_year, device, model_assemble_conf)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

model.to(device)

for epoch in range(args.epochs):
    train_loss = 0.0
    valid_loss = 0.0
    model.train()
    print("epoch:{}".format(epoch))
    for i, batch in enumerate(train_loader):
        print("batch:{}/{}, epoch:{}".format(i, len(train_loader)-1, epoch))
        optimizer.zero_grad()
        output_seq = batch[1].to(device)
        intput_ids = batch[0].to(device)
        loss, citation_pred = model(adj_list, feature_list, index_list, alignment_list, output_seq, intput_ids, train_neighbors, flag='train')
        print("current batch training loss:{}".format(loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_loss += loss.item() * args.batch_size

    train_loss = train_loss / len(train_loader.sampler)
    print("epoch: {}, training loss:{}".format(epoch, train_loss))

    print("evaluating for epoch: {}".format(epoch))
    with torch.no_grad():
        for valid_input_ids, valid_output_seq in valid_loader:
            model.eval()
            valid_input_ids = valid_input_ids.to(device)
            valid_output_seq = valid_output_seq.to(device)
            loss, val_pred = model(adj_list, feature_list, index_list, alignment_list, valid_output_seq, valid_input_ids, valid_neighbors, flag='test')
            valid_loss += loss.item() * args.val_batch_size
        valid_loss = valid_loss / len(valid_loader.sampler)
        print("epoch: {}, validation loss:{}".format(epoch, valid_loss))
        
print("training done!!")
    