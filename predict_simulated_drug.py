# %%
import torch
from torch_geometric.nn import to_hetero,GeneralConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import random
import pandas as pd
import torch_geometric.transforms as T
from sklearn.metrics import precision_score, recall_score, average_precision_score, ndcg_score

SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# %%
class Model(torch.nn.Module):
    def __init__(self, data,hidden_channels,aggr_str):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr=aggr_str)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        pred = self.decoder(z_dict, edge_label_index)
        return pred

# %%
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels)
        self.conv2 = GeneralConv((-1, -1), out_channels)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        return x

# %%
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, 2 * hidden_channels)
        self.lin2 = Linear(2 * hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[RNA_type][row], z_dict['drug'][col]], dim=-1)
        z = self.lin1(z)
        z = F.leaky_relu(z, negative_slope=0.1)
        z = self.lin2(z)
        z = F.leaky_relu(z, negative_slope=0.1)
        z = self.lin3(z)
        z = z.view(-1)
        return z

def train(train_data):
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data[RNA_type, edge_type, 'drug'].edge_label_index)
    target = train_data[RNA_type,edge_type, 'drug'].edge_label
    if torch.isnan(pred).any():
        print("NaN found in pred during training")
    loss = F.binary_cross_entropy_with_logits(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)

# %%
@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data[RNA_type, edge_type, 'drug'].edge_label_index)
    target = data[RNA_type, edge_type, 'drug'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), pred.cpu().numpy(), target.cpu().numpy()

#####################################################################
def load_lncRNA_data(sim_method='ken'):
    lncRNA_info = torch.load('./3.heter_data/lncRNA_info.pth')
    lncRNADrug_edge = lncRNA_info['lncRNADrug_edge']
    if sim_method == 'ken':
      lncRNA_edge = lncRNA_info['lncRNA_edge_ken']
    if sim_method == 'pea':
      lncRNA_edge = lncRNA_info['lncRNA_edge_pea']
    if sim_method == 'spe':
      lncRNA_edge = lncRNA_info['lncRNA_edge_spe']
    lncRNA_feature = torch.tensor(lncRNA_info['lncRNA_feature'].values, dtype=torch.float)
    lncRNA_names = lncRNA_info['lncRNA_names']
    return lncRNADrug_edge, lncRNA_edge, lncRNA_feature,lncRNA_names

def load_miRNA_data(sim_method='ken'):
    miRNA_info = torch.load('./3.heter_data/miRNA_info.pth')
    miRNADrug_edge = miRNA_info['miRNADrug_edge']
    if sim_method == 'ken':
      miRNA_edge = miRNA_info['miRNA_edge_ken']
    if sim_method == 'pea':
      miRNA_edge = miRNA_info['miRNA_edge_pea']
    if sim_method == 'spe':
      miRNA_edge = miRNA_info['miRNA_edge_spe']
    miRNA_feature = torch.tensor(miRNA_info['miRNA_feature'].values, dtype=torch.float)
    miRNA_names = miRNA_info['miRNA_names']
    return miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names

def load_drug_data():
    drug_info = torch.load('./3.heter_data/drug_info.pth')
    drug_names = drug_info['drug_names']
    drug_feature = torch.tensor(drug_info['drug_feature'].values, dtype=torch.float)
    drug_edge = drug_info['drug_edge']
    return drug_names, drug_feature, drug_edge

def load_LncMiRNA_data():
    lncRNA_miRNA_info = torch.load('./3.heter_data/LncMi_info.pth')
    LncMi_edge = lncRNA_miRNA_info['LncMi_edge']
    return LncMi_edge

# Construct heterogeneous network
def construct_heterogeneous_network(lncRNADrug_edge, lncRNA_edge, lncRNA_feature,lncRNA_names,miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names,drug_names, drug_feature, drug_edge,LncMi_edge):
    data = HeteroData()
    data['lncRNA'].node_id = torch.arange(len(lncRNA_names))
    data['miRNA'].node_id = torch.arange(len(miRNA_names))
    data['drug'].node_id = torch.arange(len(drug_names))

    data['lncRNA'].x = (lncRNA_feature - torch.min(lncRNA_feature, dim=1, keepdim=True).values) / (torch.max(lncRNA_feature, dim=1, keepdim=True).values - torch.min(lncRNA_feature, dim=1, keepdim=True).values + 1e-8)
    data['miRNA'].x = (miRNA_feature - torch.min(miRNA_feature, dim=1, keepdim=True).values) / (torch.max(miRNA_feature, dim=1, keepdim=True).values - torch.min(miRNA_feature, dim=1, keepdim=True).values)
    data['drug'].x = (drug_feature - torch.min(drug_feature, dim=1, keepdim=True).values) / (torch.max(drug_feature, dim=1, keepdim=True).values - torch.min(drug_feature, dim=1, keepdim=True).values)

    data['lncRNA', 'LncLnc', 'lncRNA'].edge_index = torch.tensor(lncRNA_edge, dtype=torch.long)
    #print("lncRNADrug_edge type:", type(lncRNADrug_edge.values))
    data['lncRNA', 'LncDrug', 'drug'].edge_index = torch.tensor(lncRNADrug_edge.values, dtype=torch.long)
    data['miRNA', 'MiMi', 'miRNA'].edge_index = torch.tensor(miRNA_edge, dtype=torch.long)
    data['miRNA', 'MiDrug', 'drug'].edge_index = torch.tensor(miRNADrug_edge.values, dtype=torch.long)
    data['drug', 'DrugDrug', 'drug'].edge_index = torch.tensor(drug_edge, dtype=torch.long)
    data['lncRNA', 'LncMi', 'miRNA'].edge_index = torch.tensor(LncMi_edge.values, dtype=torch.long)
    return data
############################################################################


def conduct_case_all_data():
    lncRNADrug_edge, lncRNA_edge, lncRNA_feature, lncRNA_names = load_lncRNA_data('ken')
    miRNADrug_edge, miRNA_edge, miRNA_feature, miRNA_names = load_miRNA_data('ken')
    drug_names, drug_feature, drug_edge = load_drug_data()
    LncMi_edge = load_LncMiRNA_data()
    print("lncRNADrug_edge shape:", lncRNADrug_edge.shape)
    # %%
    case_all_data = construct_heterogeneous_network(lncRNADrug_edge, lncRNA_edge, lncRNA_feature, lncRNA_names, miRNADrug_edge,
                                           miRNA_edge, miRNA_feature, miRNA_names, drug_names, drug_feature, drug_edge,
                                           LncMi_edge)
    # %%
    case_all_data = T.ToUndirected()(case_all_data)
    case_all_data = T.AddSelfLoops()(case_all_data)
    # %%
    case_all_data.to(device=device)
    case_all_data_file = './7.case_drug/data/case_all_data.pth'
    torch.save(case_all_data, case_all_data_file)


def construct_train_test_neg(data,edge_index_train_pos,edge_index_test_pos,drug_indice):
    positive_pairs = set(zip(edge_index_train_pos[0], edge_index_train_pos[1]))
    num_pairs = edge_index_train_pos.shape[1] * 2
    max_RNA = len(data[RNA_type].node_id)
    max_drug = len(data['drug'].node_id)
    negative_pairs = set()
    while len(negative_pairs) < num_pairs:
        RNA = np.random.randint(0, max_RNA)
        drug = np.random.randint(0, max_drug)
        if drug == drug_indice:
            continue
        if (RNA, drug) not in positive_pairs and (RNA, drug) not in negative_pairs:
            negative_pairs.add((RNA, drug))
    edge_index_train_neg = torch.tensor(list(negative_pairs), dtype=torch.long).t().to(device=device)
    test_positive_pairs = set(zip(edge_index_test_pos[0], edge_index_test_pos[1]))
    test_negative_pairs = torch.tensor([list(range(max_RNA)),[drug_indice] * max_RNA])
    test_negative_pairs_filtered = test_negative_pairs[:, ~torch.isin(test_negative_pairs[0], torch.tensor([pair[0] for pair in test_positive_pairs]))].to(device=device)
    edge_index_test_neg = test_negative_pairs_filtered
    return edge_index_train_neg,edge_index_test_neg

def conduct_case_train_test_data(RNA_type, edge_type, drug_indice):
    data = torch.load('./7.case_drug/data/case_all_data.pth')
    cols_to_remove = (data[edge_type].edge_index[1] == drug_indice).nonzero(as_tuple=True)[0].to(device=device)
    if len(cols_to_remove) != 0:
        edge_index_train_pos = data[edge_type].edge_index[:, ~torch.isin(torch.arange(data[edge_type].edge_index.size(1),device=device), cols_to_remove)]
        edge_index_train_pos_rev = edge_index_train_pos[[1, 0], :]
    else:
        edge_index_train_pos = data[edge_type].edge_index
        edge_index_train_pos_rev = data[edge_type].edge_index[[1, 0], :]
    edge_index_test_pos= data[edge_type].edge_index[:, cols_to_remove]
    pairs = list(zip(edge_index_test_pos[0].tolist(), edge_index_test_pos[1].tolist()))
    unique_pairs = list(set(pairs))
    if len(unique_pairs) < len(pairs):
        edge_index_test_pos = torch.tensor(unique_pairs, dtype=torch.long).t().to(device=device)
    edge_label_train = [1] * edge_index_train_pos.shape[1]
    edge_label_train.extend([0] * edge_index_train_pos.shape[1] * 2)
    edge_label_test = [1] * edge_index_test_pos.shape[1]
    edge_label_test.extend([0] * (len(data[RNA_type].node_id)-edge_index_test_pos.shape[1]))
    edge_index_train_neg,edge_index_test_neg = construct_train_test_neg(data, edge_index_train_pos,edge_index_test_pos, drug_indice)
    edge_index_train = torch.cat((edge_index_train_pos, edge_index_train_neg), dim=1)
    edge_index_test = torch.cat((edge_index_test_pos, edge_index_test_neg), dim=1)
    train_data = data.clone()
    train_data[edge_type].edge_index = edge_index_train_pos
    train_data['rev_' + edge_type].edge_index = edge_index_train_pos_rev
    train_data[edge_type].edge_label = torch.tensor(edge_label_train,dtype = torch.float).to(device=device)
    train_data[edge_type].edge_label_index = edge_index_train
    test_data = data.clone()
    test_data[edge_type].edge_index = edge_index_train_pos
    test_data['rev_' + edge_type].edge_index = edge_index_train_pos_rev
    test_data[edge_type].edge_label = torch.tensor(edge_label_test,dtype = torch.float).to(device=device)
    test_data[edge_type].edge_label_index = edge_index_test

    edge_index_test_np = edge_index_test.cpu().numpy()
    edge_test_df = pd.DataFrame({
        'source': edge_index_test_np[0],
        'target': edge_index_test_np[1],
        'label': edge_label_test
    })

    edge_index_train_np = edge_index_train.cpu().numpy()
    edge_train_df = pd.DataFrame({
        'source': edge_index_train_np[0],
        'target': edge_index_train_np[1],
        'label': edge_label_train
    })
    #edge_test_df.to_csv(f'./7.case_drug/data/{edge_type}_test_data_{drug_indice}.csv', index=False)
    #edge_train_df.to_csv(f'./7.case_drug/data/{edge_type}_train_data_{drug_indice}.csv', index=False)
    return train_data,test_data


def evaluate_top(test_true, test_pred,top_k):
    top_k_indices = np.argsort(test_pred)[-top_k:]
    y_true_topk = [test_true[i] for i in top_k_indices]

    precision_at_k = sum(y_true_topk) / top_k
    recall_at_k = sum(y_true_topk) / sum(test_true)
    ap = average_precision_score(test_true, test_pred)
    ndcg_at_k = ndcg_score([test_true], [test_pred], k=top_k)
    return precision_at_k,ndcg_at_k,recall_at_k,ap


if __name__ == '__main__':
    # %%
    device = torch.device('cuda')
    # %%
    # When there is no simulated novel drug dataset for the case study, it is necessary to build a dataset related to the case study. 
    # Once the dataset exists, annotate the following line and simply load it
    # conduct_case_all_data()

    # target_drug refers to the index numbers of 216 drugs, ranging from 0 to 215
    # In the benchmark dataset, there are 9 drugs that have more than 30 associations with resistance to two types of RNA, with index numbers of 0, 1, 2, 3, 4, 5, 6, 10, and 13. 
    # An example target_drug
    target_drug = 10
    top_k = 30
    RNA_type_all = [['lncRNA', 'LncDrug', target_drug], ['miRNA', 'MiDrug',target_drug]]

    for RNA_type, edge_type, drug_indice in RNA_type_all:
        print(RNA_type)
        print(edge_type)
        method_name = 'our_new'
        # load data
        train_data,test_data = conduct_case_train_test_data(RNA_type, edge_type, drug_indice)
        # %%
        if RNA_type == 'lncRNA':
           learning_rate = 0.0005
           epoch_num = 301
           channels = 64
           aggr =  'min'
        elif RNA_type == 'miRNA':
            learning_rate = 0.0001
            epoch_num = 301
            channels = 256
            aggr = 'mean'
        # %%
        model = Model(train_data, channels,aggr).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
        best_k = 0.0
        best_epoch = 0
        # %%
        for epoch in range(1, epoch_num):
            loss = train(train_data)
            train_rmse, train_pred, train_true = test(train_data)
            test_rmse, test_pred, test_true = test(test_data)
            if sum(test_true) == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {test_rmse:.4f}')
            else:
                #test_AUC = roc_auc_score(test_true, test_pred)
                #test_AUPR = average_precision_score(test_true, test_pred)
                precision_at_k, ndcg_at_k, recall_at_k,  ap = evaluate_top(test_true, test_pred, top_k)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val:{precision_at_k:.4f}, Val:{ndcg_at_k:.4f},Val:{recall_at_k:.4f},Val:{ap:.4f}')
            if (precision_at_k + ndcg_at_k)> best_k :
                best_k = precision_at_k + ndcg_at_k
                best_epoch = epoch
                res_name = f'./7.case_drug/results/{method_name}_{edge_type}_{drug_indice}_score.csv'
                edge_index_test_np = test_data[edge_type].edge_label_index.cpu().numpy()
                df = pd.DataFrame({
                   'source': edge_index_test_np[0],
                   'target': edge_index_test_np[1],
                   'y_scores': test_pred,
                   'y_true': test_true
                })
                df_sorted = df.sort_values(by='y_scores', ascending=False)
                df_sorted.to_csv(res_name, index=False)
        print('best_epoch:',best_epoch)
        print('best_PR_k:', best_k)