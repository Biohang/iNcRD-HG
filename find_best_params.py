# %%
import torch
from torch_geometric.nn import SAGEConv, to_hetero,GeneralConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import random
import pandas as pd
import itertools

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
#    pred = pred.clamp(min=0, max=5)
    target = data[RNA_type, edge_type, 'drug'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), pred.cpu().numpy(), target.cpu().numpy()


# %%
if __name__ == '__main__':
    # %%
    device = torch.device('cuda')
    method_name = 'our'
    RNA_type = 'lncRNA'
    edge_type = 'LncDrug'
    #RNA_type = 'miRNA'
    #edge_type = 'MiDrug'
    print(RNA_type)
    print(edge_type)

    learning_rate_options = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    epoch_options = [51, 101, 151, 201, 251, 301]
    hidden_channels_options = [32, 64, 128, 256, 512]
    aggr_options = ['mean', 'sum', 'max', 'min']
    param_combinations = list(
    itertools.product(learning_rate_options, epoch_options, hidden_channels_options, aggr_options))
    results = []
    # %%
    # load data
    train_data = torch.load('./3.heter_data/' + edge_type + '_train_data.pth')
    val_data = torch.load('./3.heter_data/' + edge_type + '_val_data.pth')
    test_data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
    for lr, epochs, channels, aggr in param_combinations:
        # %%
        model = Model(train_data, hidden_channels=channels,aggr_str=aggr).to(device)
        learning_rate = lr
        epoch_num = epochs
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)
        # %%
        # %%
        for epoch in range(1, epoch_num):
            loss = train(train_data)
            train_rmse, train_pred, train_true = test(train_data)
            val_rmse, val_pred, val_true = test(val_data)
        val_AUC = roc_auc_score(val_true, val_pred)
        val_AUPR = average_precision_score(val_true, val_pred)
        val_ave = (val_AUC + val_AUPR)/2.0
        results.append({
            'learning_rate': lr,
            'epochs': epochs,
            'hidden_channels': channels,
            'aggr': aggr,
            'AUC': val_AUC,
            'AUPR': val_AUPR,
            'Average':val_ave
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('./4.results/our_method_hyperparameter_search.csv', index=False)
# %%
