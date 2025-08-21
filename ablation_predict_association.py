import torch
from torch_geometric.nn import to_hetero,GeneralConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import random
import pandas as pd




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

def extract_subgraph(train_data,val_data,test_data,RNA_type):
    node_types_to_keep = [RNA_type, 'drug']
    device = train_data[RNA_type].x.device
    # Create Boolean Mask Dictionary
    subset_dict = {}
    subset_dict = {}
    for node_type in train_data.node_types:
        if node_type in node_types_to_keep:
            subset_dict[node_type] = torch.ones(train_data[node_type].x.size(0), dtype=torch.bool, device=device)
        else:
            subset_dict[node_type] = torch.zeros(train_data[node_type].x.size(0), dtype=torch.bool, device=device)
    # Remove miRNA/lncRNA nodes and related edges
    train_data_new = train_data.subgraph(subset_dict)
    val_data_new = val_data.subgraph(subset_dict)
    test_data_new = test_data.subgraph(subset_dict)
    return train_data_new, val_data_new, test_data_new
# %%

def node_onehot(train_data_all, val_data_all, test_data_all):
    num_lncRNA = train_data_all['lncRNA'].x.size(0) 
    num_miRNA = train_data_all['miRNA'].x.size(0) 
    num_drug = train_data_all['drug'].x.size(0) 
    # Generate one hot encoding
    lncRNA_one_hot = F.one_hot(torch.arange(num_lncRNA), num_classes=num_lncRNA).float().to(train_data_all['lncRNA'].x.device)
    miRNA_one_hot = F.one_hot(torch.arange(num_miRNA), num_classes=num_miRNA).float().to(train_data_all['miRNA'].x.device)
    drug_one_hot = F.one_hot(torch.arange(num_drug), num_classes=num_drug).float().to(train_data_all['drug'].x.device)
    # Replace original node features
    train_data_all['lncRNA'].x = lncRNA_one_hot
    train_data_all['miRNA'].x = miRNA_one_hot
    train_data_all['drug'].x = drug_one_hot
    val_data_all['lncRNA'].x = lncRNA_one_hot
    val_data_all['miRNA'].x = miRNA_one_hot
    val_data_all['drug'].x = drug_one_hot
    test_data_all['lncRNA'].x = lncRNA_one_hot
    test_data_all['miRNA'].x = miRNA_one_hot
    test_data_all['drug'].x = drug_one_hot
    return train_data_all,val_data_all,test_data_all


# %%
if __name__ == '__main__':
    # %%
    device = torch.device('cuda')
    # %%
    RNA_type_all = [['lncRNA', 'LncDrug'], ['miRNA', 'MiDrug']]

    for RNA_type, edge_type in RNA_type_all:
        # Different ablation strategies
        method_name = f'wo_{RNA_type}'
        #method_name = 'wo_attribute'
        #method_name = f'wo_attribute_{RNA_type}'
        # Load data
        train_data_all = torch.load('./3.heter_data/' + edge_type + '_train_data.pth')
        val_data_all = torch.load('./3.heter_data/' + edge_type + '_val_data.pth')
        test_data_all = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')

        # wo_{RNA_type}: For the LncDrug/MiDrug task, extract subgraphs (excluding intermediate nodes/edges containing intermediate nodes) 
        train_data,val_data, test_data = extract_subgraph(train_data_all,val_data_all,test_data_all,RNA_type)

        # wo_attribute: Replace original RNA expression features and drug SMILES features with one hot encoding features
        #train_data, val_data, test_data = node_onehot(train_data_all, val_data_all, test_data_all)

        # wo_attribute_{RNA_type} 
        #train_data, val_data, test_data = node_onehot(train_data_all, val_data_all, test_data_all)
        #train_data,val_data, test_data = extract_subgraph(train_data,val_data,test_data,RNA_type)

        # %%
        if RNA_type == 'lncRNA':
           learning_rate = 0.0005
           epoch_num = 101
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
        # %%
        for epoch in range(1, epoch_num):
            loss = train(train_data)
            train_rmse, train_pred, train_true = test(train_data)
            #test_rmse, test_pred, test_true = test(test_data)
            test_rmse, test_pred, test_true = test(val_data)
            test_AUC = roc_auc_score(test_true, test_pred)
            test_AUPR = average_precision_score(test_true, test_pred)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {test_rmse:.4f}, Val:{test_AUC:.4f}, Val:{test_AUPR:.4f}')
            #res_name = f'./6.ablation_results/test_{method_name}_{edge_type}_score.csv'
            res_name = f'./6.ablation_results/val_{method_name}_{edge_type}_score.csv'
            df = pd.DataFrame({
                'y_scores': test_pred, 
                'y_true': test_true 
            })
            df.to_csv(res_name, index=False)
       # %%