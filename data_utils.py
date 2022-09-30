import torch
import networkx as nx
import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split

import pdb


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def generate_labels(data_dir, pub_year, time_steps, sample_names, name2idx):
    with open(data_dir+'/citation_label_'+str(time_steps)+'_' + str(pub_year), 'rb') as f:
        tmp = pickle.load(f)
    sample_idx = [name2idx[name] for name in sample_names]
    citations_seqs = [tmp[name] for name in sample_names]
    citation_seqs_label = []
    for seq in citations_seqs:
        seq_label = []
        for item in seq:
            seq_label.append(list(item.values())[0])
        citation_seqs_label.append(seq_label)
    labels = np.array(citation_seqs_label)
    # max_cite = labels.max()
    # min_cite = labels.min()
    # norm_seqs = (labels - min_cite) / (max_cite - min_cite)
    # new_seqs = np.where(labels > 0, np.log10(labels)+1, 0)
    train_ids, test_ids, train_seqs, test_seqs = train_test_split(sample_idx, labels, test_size=0.33, random_state=42)
    return train_ids, test_ids, train_seqs, test_seqs

def generate_relation_index(graph_now, sample_idx, name_idx, rel_type):
    relation_dict = {}
    idx_name = graph_now['idx_name']
    adj = graph_now['adj']
    for idx in sample_idx:
        neighbors = [idx_name[n] for n in adj[rel_type].neighbors(idx)]
        patent_neighbors = []
        for item in neighbors:
            if str(item) in name_idx:
                patent_neighbors.append(name_idx[str(item)])
        relation_dict[idx] = patent_neighbors
    return relation_dict

def get_neighbors(data_dir, idx_list, pub_year, rel_types):
    res_neighbors = {}
    with open(data_dir + '/temporal_graph_' + str(pub_year), 'rb') as f:
        graph = pickle.load(f)
        adj = graph['adj']
        for idx in idx_list:
            all_neighbors = []
            for rel in rel_types:
                neighbors = [n for n in adj[rel].neighbors(idx)]
                all_neighbors.append(neighbors)
            res_neighbors[idx] = all_neighbors
    return res_neighbors

def get_graph_label(data_dir, pub_year, time_steps_history, time_steps_predict, subtask):
    print('Loading data...')
    rel_types = []
    adj_list = []
    feature_list = []
    index_list = []
    index_name = []
    name_index = []
    name2idx = None
    alignment_list = None
    sample_names = None
    graph_now = None
    
    with open(data_dir+'/sample_names_'+str(pub_year)+'_'+subtask, 'r') as f:
    # with open(data_dir+'/sample_names_'+str(pub_year)+'_newborn', 'r') as f:
    # with open(data_dir+'/sample_names_'+str(pub_year)+'_grown', 'r') as f:
        content = f.readlines()
        sample_names = [name.strip() for name in content]

    with open(data_dir + '/temporal_graph_' + str(pub_year), 'rb') as f:
        graph_now = pickle.load(f)
        name2idx = graph_now['name_idx']
        sample_idx = [name2idx[name] for name in sample_names]
        rel_types = list(graph_now['adj'].keys())

    with open(data_dir + '/alignment_list_'+str(pub_year)+'_'+str(time_steps_history), 'rb') as f:
        alignment_list = torch.Tensor(pickle.load(f)).type(torch.int64)

    for i in range(time_steps_history):
        with open(data_dir+'/temporal_graph_' + str(pub_year-time_steps_history+i), 'rb') as f:
            tmp = pickle.load(f)
        adj = []
        index = []
        for rel_type in rel_types:
            adj_sparse_matrix = nx.to_scipy_sparse_matrix(tmp['adj'][rel_type])
            adj_sparse_tensor = sparse_mx_to_torch_sparse_tensor(adj_sparse_matrix)
            adj.append(adj_sparse_tensor)
            rel_index = generate_relation_index(graph_now, sample_idx, tmp['name_idx'], rel_type)
            index.append(rel_index)

        adj_list.append(adj) # years(list) * relation types(list) * nodes(list) * nodes(list)
        feature_list.append(torch.from_numpy(tmp['feature'])) # years(list) * patents(list) * dim(list)
        index_list.append(index) # years(list) * relation types(list) * patents(dict) * relations(list)
        index_name.append(tmp['idx_name']) # years(list) * idx_name(dict)
        name_index.append(tmp['name_idx']) # years(list) * name_idx(dict)


    train_ids, test_ids, train_seqs, test_seqs = generate_labels(data_dir, pub_year, time_steps_predict, sample_names, name2idx)
    labels = {'train_ids': train_ids, 'test_ids': test_ids, 'train_seqs': train_seqs, 'test_seqs': test_seqs}
    print('Data load complete!')

    return adj_list, feature_list, index_list, alignment_list, labels, rel_types, index_name, name_index


    