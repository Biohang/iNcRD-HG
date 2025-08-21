import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
import numpy as np
import torch_geometric.nn as pyg_nn
import random
import pandas as pd

# %%
SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



class GNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNDecoder, self).__init__()
        # Basic graph encoders compared to iNcRD-HG
        #self.gcn1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.GATConv(in_channels, hidden_channels,heads=2)
        #self.gcn2 = pyg_nn.GATConv(2*hidden_channels, out_channels,heads=2)
        self.gcn1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        self.gcn2 = pyg_nn.SAGEConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.LEConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.LEConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.GeneralConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.GeneralConv(hidden_channels, out_channels)
        #self.gcn1 = pyg_nn.GraphConv(in_channels, hidden_channels)
        #self.gcn2 = pyg_nn.GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)

        num_RNA = x.shape[0] - 216
        print('num_RNA:',num_RNA)
        RNA_emb = x[:num_RNA]
        drug_emb = x[num_RNA:]

        # Decoder: By dot multiplication
        out = torch.matmul(RNA_emb, drug_emb.T)
        return out


def train(model, data, optimizer, train_edge_label_index, train_edge_label):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index) 
    out = out[train_edge_label_index[0], train_edge_label_index[1]]
    labels = train_edge_label
    loss = nn.BCEWithLogitsLoss()(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, test_edge_label_index, test_edge_label,method_name,target_drug,top_k):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        out = out[test_edge_label_index[0], test_edge_label_index[1]]
        labels = test_edge_label

        # Calculate AUC and AUPR
        y_true = labels.cpu().numpy()
        y_scores = out.cpu().numpy()
        df = pd.DataFrame({
            'y_scores': y_scores, 
            'y_true': y_true 
        })
        res_name = f'./7.case_drug/compare_results/{method_name}_{edge_type}_{target_drug}_score.csv'
        df.to_csv( res_name, index=False)
        precision_at_k, ndcg_at_k, recall_at_k,  ap = evaluate_top(y_true, y_scores, top_k)
    return precision_at_k, ndcg_at_k, recall_at_k,  ap


def edge_index_to_matrix(edge_index, num_code_1, num_code_2):
    matrix = torch.zeros((num_code_1, num_code_2), dtype=torch.float32)
    row, col = edge_index
    matrix[row, col] = 1
    matrix[col, row] = 1
    return matrix

def construct_train_test_neg(data_all,edge_index_train_pos,edge_index_test_pos,drug_indice):
    positive_pairs = set(zip(edge_index_train_pos[0], edge_index_train_pos[1]))
    num_pairs = edge_index_train_pos.shape[1] * 2
    max_RNA = len(data_all[node_type].node_id)
    max_drug = len(data_all['drug'].node_id)
    negative_pairs = set()
    while len(negative_pairs) < num_pairs:
        RNA = np.random.randint(0, max_RNA)
        drug = np.random.randint(0, max_drug)
        if drug == drug_indice:
            continue
        if (RNA, drug) not in positive_pairs and (RNA, drug) not in negative_pairs:
            negative_pairs.add((RNA, drug))
    edge_index_train_neg = torch.tensor(list(negative_pairs), dtype=torch.long).t().to(device='cuda:0')
    test_positive_pairs = set(zip(edge_index_test_pos[0], edge_index_test_pos[1]))
    # Generate initial test_negative_pairs
    test_negative_pairs = torch.tensor([list(range(max_RNA)),[drug_indice] * max_RNA])
    # Ensure that generated test_negative_pairs do not contain any pairs in test_positive_pairs
    test_negative_pairs_filtered = test_negative_pairs[:, ~torch.isin(test_negative_pairs[0], torch.tensor([pair[0] for pair in test_positive_pairs]))].to(device='cuda:0')
    edge_index_test_neg = test_negative_pairs_filtered
    return edge_index_train_neg,edge_index_test_neg

def conduct_case_train_test_data(RNA_type, edge_type, target_drug):
    data_all = torch.load('./7.case_drug/data/case_all_data.pth')
    cols_to_remove = (data_all[edge_type].edge_index[1] == target_drug).nonzero(as_tuple=True)[0].to(device='cuda:0')
    if len(cols_to_remove) != 0:
        edge_index_train_pos = data_all[edge_type].edge_index[:, ~torch.isin(torch.arange(data_all[edge_type].edge_index.size(1),device='cuda:0'), cols_to_remove)]
        edge_index_train_pos_rev = edge_index_train_pos[[1, 0], :]
    else:
        edge_index_train_pos = data_all[edge_type].edge_index
        edge_index_train_pos_rev = data_all[edge_type].edge_index[[1, 0], :]
    edge_index_test_pos= data_all[edge_type].edge_index[:, cols_to_remove]
    pairs = list(zip(edge_index_test_pos[0].tolist(), edge_index_test_pos[1].tolist()))
    unique_pairs = list(set(pairs))
    if len(unique_pairs) < len(pairs):
        edge_index_test_pos = torch.tensor(unique_pairs, dtype=torch.long).t().to(device='cuda:0')
    edge_label_train = [1] * edge_index_train_pos.shape[1]
    edge_label_train.extend([0] * edge_index_train_pos.shape[1] * 2)
    edge_label_test = [1] * edge_index_test_pos.shape[1]
    edge_label_test.extend([0] * (len(data_all[RNA_type].node_id)-edge_index_test_pos.shape[1]))
    edge_index_train_neg,edge_index_test_neg = construct_train_test_neg(data_all, edge_index_train_pos,edge_index_test_pos, target_drug)
    edge_index_train = torch.cat((edge_index_train_pos, edge_index_train_neg), dim=1)
    edge_index_test = torch.cat((edge_index_test_pos, edge_index_test_neg), dim=1)
    train_data = data_all.clone()
    train_data[edge_type].edge_index = edge_index_train_pos
    train_data['rev_' + edge_type].edge_index = edge_index_train_pos_rev
    train_data[edge_type].edge_label = torch.tensor(edge_label_train,dtype = torch.float).to(device='cuda:0')
    train_data[edge_type].edge_label_index = edge_index_train
    test_data = data_all.clone()
    test_data[edge_type].edge_index = edge_index_train_pos
    test_data['rev_' + edge_type].edge_index = edge_index_train_pos_rev
    test_data[edge_type].edge_label = torch.tensor(edge_label_test,dtype = torch.float).to(device='cuda:0')
    test_data[edge_type].edge_label_index = edge_index_test
    return train_data,test_data

def evaluate_top(test_true, test_pred,top_k):
    top_k_indices = np.argsort(test_pred)[-top_k:]
    y_true_topk = [test_true[i] for i in top_k_indices]

    # Calculate four evaluation indicators that focus on the top k predicted results
    precision_at_k = sum(y_true_topk) / top_k
    recall_at_k = sum(y_true_topk) / sum(test_true)
    ap = average_precision_score(test_true, test_pred)
    ndcg_at_k = ndcg_score([test_true], [test_pred], k=top_k)
    return precision_at_k,ndcg_at_k,recall_at_k,ap



def predict_target_data(method_name, learn_rate,edge_type,node_type,sim_type,target_drug,top_k):
    train_data,test_data = conduct_case_train_test_data(node_type, edge_type, target_drug)
    test_data_new = test_data.clone()
    num_RNA = train_data[node_type].x.shape[0]
    num_drug = train_data['drug'].x.shape[0]
    num_nodes = num_RNA + num_drug
    RNA_features = train_data[node_type].x
    drug_features = train_data['drug'].x

    # Construct node feature matrix, taking lncRNA-drug as an example (lncRNA+drug) ->(1322+216, 938+768)
    node_features = torch.zeros((num_nodes, RNA_features.shape[1]+drug_features.shape[1]), dtype=torch.float32)
    node_features[:num_RNA, :RNA_features.shape[1]] = RNA_features
    node_features[num_RNA:, RNA_features.shape[1]:] = drug_features
    node_features  = node_features.to('cuda:0')

    # Construct edges in the graph (this work also includes RNA direct similar edges and drug direct similar edges, as well as RNA drug interaction edges)
    # Due to the fact that RNA and drugs were originally indexed from 0, they are now considered to be of the same type, and the drug index needs to be adjusted
    edge_index_RNA_sim = train_data[node_type,sim_type,node_type].edge_index
    edge_index_drug_sim = train_data['drug','DrugDrug','drug'].edge_index
    edge_index_drug_sim[0] += num_RNA
    edge_index_drug_sim[1] += num_RNA
    edge_index_train = train_data[node_type,edge_type,'drug'].edge_index
    edge_index_train[1] += num_RNA

    edge_index_test = test_data_new[node_type,edge_type,'drug'].edge_index
    edge_index_test[1] += num_RNA
    reverse_edge_index_RNA_sim = edge_index_RNA_sim.flip(0)
    reverse_edge_index_drug_sim = edge_index_drug_sim.flip(0)
    reverse_edge_index_train = edge_index_train.flip(0)
    reverse_edge_index_test = edge_index_test.flip(0)
    edge_index_RNA_sim_combined = torch.cat([edge_index_RNA_sim, reverse_edge_index_RNA_sim], dim=1)
    edge_index_drug_sim_combined = torch.cat([edge_index_drug_sim, reverse_edge_index_drug_sim], dim=1)
    edge_index_combined_train = torch.cat([edge_index_train, reverse_edge_index_train,edge_index_RNA_sim_combined,edge_index_drug_sim_combined], dim=1)

    edge_index_combined_test = torch.cat([edge_index_test, reverse_edge_index_test,edge_index_RNA_sim_combined,edge_index_drug_sim_combined], dim=1)

    # Build positive and negative sample indices and labels in the training set
    train_edge_label_index = train_data[node_type,edge_type,'drug'].edge_label_index
    train_edge_label = train_data[node_type,edge_type,'drug'].edge_label

    # Build positive and negative sample indices and labels in the test set
    test_edge_label_index = test_data[node_type,edge_type,'drug'].edge_label_index
    test_edge_label = test_data[node_type,edge_type,'drug'].edge_label

    # Build graph data objects
    data_train = Data(x=node_features, edge_index=edge_index_combined_train)
    data_test = Data(x=node_features, edge_index=edge_index_combined_test)
    # Define models and optimizers
    model = GNNDecoder(in_channels=node_features.shape[1], hidden_channels=128, out_channels=128).to('cuda:0')
    optimizer = optim.RMSprop(model.parameters(), lr=learn_rate, alpha=0.99)

    # Train model
    num_epochs = 200
    for epoch in range(num_epochs):
        loss = train(model, data_train, optimizer, train_edge_label_index, train_edge_label)

        if epoch % 40 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Test model performance
        precision_at_k, ndcg_at_k, recall_at_k,  ap = evaluate(model, data_test, test_edge_label_index, test_edge_label,method_name,target_drug,top_k)
        #print(f'precision_at_k: {precision_at_k}, ndcg_at_k: {ndcg_at_k}, recall_at_k: {recall_at_k}, ap: {ap}')
    return precision_at_k, ndcg_at_k,recall_at_k,ap


method_name = 'SAGEConv'
RNA_type_all = [['lncRNA', 'LncDrug', 'LncLnc', 0.0001], ['miRNA', 'MiDrug','MiMi',0.0001]]
#In the benchmark dataset, there are 9 drugs that have more than 30 associations with resistance to two types of RNA, with index numbers of 0, 1, 2, 3, 4, 5, 6, 10, and 13. 
target_drug_list = [0,1,2,3,4,5,6,10,13]
top_k = 30

# The evaluation process focuses on the performance of the top prediction results
for node_type, edge_type, sim_type,learn_rate in RNA_type_all:
    precision_list = []
    ndcg_list = []
    recall_list = []
    ap_list = []
    for target_drug in target_drug_list:
        precision_at_k, ndcg_at_k, recall_at_k, ap = predict_target_data(method_name, learn_rate,edge_type,node_type,sim_type,target_drug,top_k)
        precision_list.append(precision_at_k)
        ndcg_list.append(ndcg_at_k)
        recall_list.append(recall_at_k)
        ap_list.append(ap)
    df = pd.DataFrame({
        'target_drug': target_drug_list,
        'precision_at_k': precision_list,
        'ndcg_at_k': ndcg_list,
        'recall_at_k': recall_list,
        'ap': ap_list
        })
    df.to_csv(f'7.case_drug/compare_results/{edge_type}_performance_{method_name}.csv', index=False)





