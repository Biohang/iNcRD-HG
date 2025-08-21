import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import torch

#data_type = 'test'
data_type = 'val'
edge_type = 'LncDrug'
#edge_type = 'MiDrug'

data = torch.load('./3.heter_data/' + edge_type + '_' + data_type + '_data.pth')
data_label = data[edge_type].edge_label.cpu().numpy()
data_index = data[edge_type].edge_label_index.cpu().numpy()
pos_index = np.where(data_label == 1)[0]
neg_index = np.where(data_label == 0)[0]
RNA_pos_index = data_index[0,pos_index]
drug_pos_index = data_index[1,pos_index]
RNA_neg_index = data_index[0,neg_index]
drug_neg_index = data_index[1,neg_index]

#####################################################################################
# Build node attribute features and visualize
drug_feature = pd.read_csv('./2.construct_network/drug_features-chemBertA.csv')
lncRNA_feature = pd.read_csv('./2.construct_network/LncExp.csv')
miRNA_feature = pd.read_csv('./2.construct_network/MiExp.csv')
drug_similarity = pd.read_csv('./2.construct_network/drugSimMat.csv',header=None)
lncRNA_similarity = pd.read_csv('./2.construct_network/LncSimMat_ken.csv',header=None)
miRNA_similarity = pd.read_csv('./2.construct_network/MiSimMat_ken.csv',header=None)

if edge_type=='LncDrug':
    RNA_feature_new = lncRNA_feature.iloc[:, 1:].to_numpy().T
else:
    RNA_feature_new = miRNA_feature.iloc[:, 1:].to_numpy().T

drug_feature_new = drug_feature.iloc[:, 1:].to_numpy()
RNA_pos_feature=RNA_feature_new[RNA_pos_index,:]
drug_pos_feature=drug_feature_new[drug_pos_index,:]
pos_feature=np.hstack((RNA_pos_feature,drug_pos_feature))

RNA_neg_feature=RNA_feature_new[RNA_neg_index,:]
drug_neg_feature=drug_feature_new[drug_neg_index,:]
neg_feature=np.hstack((RNA_neg_feature,drug_neg_feature))
feature_all = np.vstack((pos_feature,neg_feature))
feature_all_new = TSNE(n_components=3,init='pca').fit_transform(feature_all)
x_red = feature_all_new[0:pos_feature.shape[0], 0]
y_red = feature_all_new[0:pos_feature.shape[0], 1]
z_red = feature_all_new[0:pos_feature.shape[0], 2]
x_blue = feature_all_new[pos_feature.shape[0]:feature_all.shape[0], 0]
y_blue = feature_all_new[pos_feature.shape[0]:feature_all.shape[0], 1]
z_blue = feature_all_new[pos_feature.shape[0]:feature_all.shape[0], 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if edge_type=='LncDrug':
   ax.scatter(x_red, y_red, z_red,  c='#299D8F',alpha=0.6, s=5)
   ax.scatter(x_blue, y_blue, z_blue, c='#E7C66B',alpha=0.6, s=5)
else:
   ax.scatter(x_red, y_red,z_red, c='#f3a361', alpha=0.6, s=5)
   ax.scatter(x_blue, y_blue,z_blue, c='#669aba', alpha=0.6, s=5)
ax.view_init(elev=30, azim=-60)
plt.savefig(f'./8.feature_visualization/attribute_feature_{edge_type}_{data_type}.jpg', dpi=900)

#####################################################################################
# Construct node structural features and visualize
if edge_type=='LncDrug':
    RNA_feature_new = lncRNA_similarity.iloc[:, 0:].to_numpy()
else:
    RNA_feature_new = miRNA_similarity.iloc[:, 0:].to_numpy()

nan_indices = np.isnan(RNA_feature_new)
RNA_feature_new[nan_indices] = 0
drug_feature_new = drug_feature.iloc[:, 1:].to_numpy()
RNA_pos_feature = RNA_feature_new[RNA_pos_index,:]
drug_pos_feature = drug_feature_new[drug_pos_index,:]
pos_feature=np.hstack((RNA_pos_feature,drug_pos_feature))

RNA_neg_feature=RNA_feature_new[RNA_neg_index,:]
drug_neg_feature=drug_feature_new[drug_neg_index,:]
neg_feature=np.hstack((RNA_neg_feature,drug_neg_feature))
feature_all = np.vstack((pos_feature,neg_feature))
feature_all_new = TSNE(n_components=3,init='pca').fit_transform(feature_all)
x_red = feature_all_new[0:pos_feature.shape[0], 0]
y_red = feature_all_new[0:pos_feature.shape[0], 1]
z_red = feature_all_new[0:pos_feature.shape[0], 2]
x_blue = feature_all_new[pos_feature.shape[0]:feature_all.shape[0], 0]
y_blue = feature_all_new[pos_feature.shape[0]:feature_all.shape[0], 1]
z_blue = feature_all_new[pos_feature.shape[0]:feature_all.shape[0], 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if edge_type=='LncDrug':
   ax.scatter(x_red, y_red, z_red,  c='#299D8F',alpha=0.6, s=5)
   ax.scatter(x_blue, y_blue, z_blue, c='#E7C66B',alpha=0.6, s=5)
else:
   ax.scatter(x_red, y_red,z_red, c='#f3a361', alpha=0.6, s=5)
   ax.scatter(x_blue, y_blue,z_blue, c='#669aba', alpha=0.6, s=5)
ax.view_init(elev=30, azim=-60)
plt.savefig(f'./8.feature_visualization/structure_feature_{edge_type}_{data_type}.jpg', dpi=900)


#####################################################################################
# Extract deep features learned by iNcRD-HG and visualize
latent_feature = np.load(f"./8.feature_visualization/latent_feature_{edge_type}_{data_type}.npy")
feature_all_new = TSNE(n_components=3,init='pca').fit_transform(latent_feature)
x_red = feature_all_new[pos_index[0:len(pos_index)], 0]
y_red = feature_all_new[pos_index[0:len(pos_index)], 1]
z_red = feature_all_new[pos_index[0:len(pos_index)], 2]
x_blue = feature_all_new[neg_index[0:len(neg_index)], 0]
y_blue = feature_all_new[neg_index[0:len(neg_index)], 1]
z_blue = feature_all_new[neg_index[0:len(neg_index)], 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if edge_type=='LncDrug':
   ax.scatter(x_red, y_red, z_red,  c='#299D8F',alpha=0.6, s=5)
   ax.scatter(x_blue, y_blue, z_blue, c='#E7C66B',alpha=0.6,s=5)
else:
   ax.scatter(x_red, y_red,z_red, c='#f3a361', alpha=0.6,s=5)
   ax.scatter(x_blue, y_blue,z_blue, c='#669aba', alpha=0.6,s=5)
ax.view_init(elev=30, azim=-60)
plt.savefig(f'./8.feature_visualization/latent_feature_{edge_type}_{data_type}.jpg', dpi=900)
