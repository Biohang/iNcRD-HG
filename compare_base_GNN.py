import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
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
        # Different grah encoders
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

        # Obtain the embeddings of RNA and drug (splicing)
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


def evaluate(model, data, test_edge_label_index, test_edge_label,method_name):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

        out = out[test_edge_label_index[0], test_edge_label_index[1]]
        labels = test_edge_label

        y_true = labels.cpu().numpy()
        y_scores = out.cpu().numpy()
        df = pd.DataFrame({
            'y_scores': y_scores,
            'y_true': y_true 
        })
        res_name = f'./4.results/{method_name}_{edge_type}_score.csv'
        df.to_csv( res_name, index=False)
        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
    return auc, aupr


def edge_index_to_matrix(edge_index, num_code_1, num_code_2):
    matrix = torch.zeros((num_code_1, num_code_2), dtype=torch.float32)
    row, col = edge_index
    matrix[row, col] = 1
    matrix[col, row] = 1
    return matrix


learn_rate=0.0005
#edge_type = 'LncDrug'
#node_type = 'lncRNA'
#sim_type = 'LncLnc'
edge_type = 'MiDrug'
node_type = 'miRNA'
sim_type = 'MiMi'
method_name = 'SAGE'

train_data = torch.load('./3.heter_data/' + edge_type + '_train_data.pth')
test_data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
num_RNA = train_data[node_type].x.shape[0]
num_drug = train_data['drug'].x.shape[0]
num_nodes = num_RNA + num_drug
RNA_features = train_data[node_type].x
drug_features = train_data['drug'].x
# Replace original RNA expression and drug SMILES features with one hot encoding for RNA_feature and drug_feature
#RNA_labels = torch.arange(num_RNA)
#RNA_features = torch.nn.functional.one_hot(RNA_labels, num_classes=num_RNA).to('cuda:0')
#drug_labels = torch.arange(num_drug)
#drug_features = torch.nn.functional.one_hot(drug_labels, num_classes=num_drug).to('cuda:0')

# Construct node feature matrix, taking lncRNA-drug as an example (lncRNA+drug) ->(1322+216, 938+768)
node_features = torch.zeros((num_nodes, RNA_features.shape[1]+drug_features.shape[1]), dtype=torch.float32)
node_features[:num_RNA, :RNA_features.shape[1]] = RNA_features
node_features[num_RNA:, RNA_features.shape[1]:] = drug_features
node_features  = node_features.to('cuda:0')
print('node_features:',node_features.device)


# Construct edges in the graph (this work also includes RNA direct similar edges and drug direct similar edges, as well as RNA drug interaction edges)
# Due to the fact that RNA and drugs were originally indexed from 0, they are now considered to be of the same type, and the drug index needs to be adjusted
edge_index_RNA_sim = train_data[node_type,sim_type,node_type].edge_index
edge_index_drug_sim = train_data['drug','DrugDrug','drug'].edge_index
edge_index_drug_sim[0] += num_RNA
edge_index_drug_sim[1] += num_RNA
edge_index_train = train_data[node_type,edge_type,'drug'].edge_index
edge_index_train[1] += num_RNA
edge_index_test = test_data[node_type,edge_type,'drug'].edge_index
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
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Test model performance
auc, aupr = evaluate(model, data_test, test_edge_label_index, test_edge_label,method_name)
print(f'Test AUC: {auc}, AUPR: {aupr}')
