import pandas as pd
import numpy as np
import torch
import argparse

def convert_to_binary(value):
    return 1 if value > 0 else 0

def modify_node_index(index_value,lncRNA_size,miRNA_size,drug_size):
    if index_value < lncRNA_size:
        return index_value
    elif lncRNA_size <= index_value < (lncRNA_size + miRNA_size):
        return index_value - lncRNA_size
    elif (lncRNA_size + miRNA_size) <= index_value < (lncRNA_size + miRNA_size + drug_size):
        return index_value - (lncRNA_size + miRNA_size)
    else:
        return index_value


# Generate all edges of the graph structure in test_data/val_data and concatenate them
def save_edge_index_all(edge_type, file_type):
    if file_type == 'test':
        data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
    else:
        data = torch.load('./3.heter_data/' + edge_type + '_val_data.pth')

    LncMi_edge = data['LncMi'].edge_index.T.cpu().detach().numpy()
    LncMi_edge_new = pd.DataFrame(LncMi_edge, columns=['index_0', 'index_1'])
    LncMi_edge_new['node_0'] = 'lncRNA'
    LncMi_edge_new['node_1'] = 'miRNA'

    LncDrug_edge = data['LncDrug'].edge_index.T.cpu().detach().numpy()
    LncDrug_edge_new = pd.DataFrame(LncDrug_edge, columns=['index_0', 'index_1'])
    LncDrug_edge_new['node_0'] = 'lncRNA'
    LncDrug_edge_new['node_1'] = 'drug'

    MiDrug_edge = data['MiDrug'].edge_index.T.cpu().detach().numpy()
    MiDrug_edge_new = pd.DataFrame(MiDrug_edge, columns=['index_0', 'index_1'])
    MiDrug_edge_new['node_0'] = 'miRNA'
    MiDrug_edge_new['node_1'] = 'drug'

    LncLnc_edge = data['LncLnc'].edge_index.T.cpu().detach().numpy()
    LncLnc_edge_new = pd.DataFrame(LncLnc_edge, columns=['index_0', 'index_1'])
    LncLnc_edge_new['node_0'] = 'lncRNA'
    LncLnc_edge_new['node_1'] = 'lncRNA'

    MiMi_edge = data['MiMi'].edge_index.T.cpu().detach().numpy()
    MiMi_edge_new = pd.DataFrame(MiMi_edge, columns=['index_0', 'index_1'])
    MiMi_edge_new['node_0'] = 'miRNA'
    MiMi_edge_new['node_1'] = 'miRNA'

    DrugDrug_edge = data['DrugDrug'].edge_index.T.cpu().detach().numpy()
    DrugDrug_edge_new = pd.DataFrame(DrugDrug_edge, columns=['index_0', 'index_1'])
    DrugDrug_edge_new['node_0'] = 'drug'
    DrugDrug_edge_new['node_1'] = 'drug'

    edge_index_all = pd.concat([LncMi_edge_new, LncDrug_edge_new, MiDrug_edge_new,LncLnc_edge_new,MiMi_edge_new,DrugDrug_edge_new], axis=0,ignore_index=True)

    return edge_index_all


# Calculate contribution ratio of three different types of nodes for predicting lncRNA-drug/miRNA-drug resistance association
def explain_node(edge_type,file_type):
    if file_type == 'test':
        path = './5.graph_important/'
    else:
        path = './5.graph_important/val_'
    drug_contributions = pd.read_csv(f'{path}drug_importance_for_test_{edge_type}.csv')
    lncRNA_contributions = pd.read_csv(f'{path}lncRNA_importance_for_test_{edge_type}.csv')
    miRNA_contributions = pd.read_csv(f'{path}miRNA_importance_for_test_{edge_type}.csv')

    # Transform all contribution matrices by assigning 1 to those with>0, while assigning 0 to others
    drug_contributions_bool = drug_contributions.applymap(convert_to_binary)
    lncRNA_contributions_bool = lncRNA_contributions.applymap(convert_to_binary)
    miRNA_contributions_bool = miRNA_contributions.applymap(convert_to_binary)

    drug_avg_contributions = np.mean(drug_contributions_bool.mean(axis=0))
    lncRNA_avg_contributions =  np.mean(lncRNA_contributions_bool.mean(axis=0))
    miRNA_avg_contributions =  np.mean(miRNA_contributions_bool.mean(axis=0))

    # Calculate the total contribution
    total_contribution = drug_avg_contributions + miRNA_avg_contributions + lncRNA_avg_contributions
    # Calculate the contribution ratio of each node type
    drug_proportion = drug_avg_contributions / total_contribution
    miRNA_proportion = miRNA_avg_contributions / total_contribution
    lncRNA_proportion = lncRNA_avg_contributions / total_contribution
# %%
    print('Node importance for test ', edge_type)
    print('drug_proportion:',drug_proportion)
    print('miRNA_proportion:', miRNA_proportion)
    print('lncRNA_proportion:', lncRNA_proportion)


# %%
# Calculate contribution ratio of different types of edges for predicting lncRNA-drug/miRNA-drug resistance association
def explain_edge(edge_type,file_type):
    if file_type == 'test':
        path = './5.graph_important/'
    else:
        path = './5.graph_important/val_'
    LncMi_contributions = pd.read_csv(f'{path}(\'lncRNA\', \'LncMi\', \'miRNA\')_importance_for_test_{edge_type}.csv')
    DrugDrug_contributions = pd.read_csv(f'{path}(\'drug\', \'DrugDrug\', \'drug\')_importance_for_test_{edge_type}.csv')
    MiDrug_contributions = pd.read_csv(f'{path}(\'miRNA\', \'MiDrug\', \'drug\')_importance_for_test_{edge_type}.csv')
    MiMi_contributions = pd.read_csv(f'{path}(\'miRNA\', \'MiMi\', \'miRNA\')_importance_for_test_{edge_type}.csv')
    LncDrug_contributions = pd.read_csv(f'{path}(\'lncRNA\', \'LncDrug\', \'drug\')_importance_for_test_{edge_type}.csv')
    LncLnc_contributions = pd.read_csv(f'{path}(\'lncRNA\', \'LncLnc\', \'lncRNA\')_importance_for_test_{edge_type}.csv')
    rev_LncMi_contributions = pd.read_csv(f'{path}(\'miRNA\', \'rev_LncMi\', \'lncRNA\')_importance_for_test_{edge_type}.csv')
    rev_LncDrug_contributions = pd.read_csv(f'{path}(\'drug\', \'rev_LncDrug\', \'lncRNA\')_importance_for_test_{edge_type}.csv')
    rev_MiDrug_contributions = pd.read_csv(f'{path}(\'drug\', \'rev_MiDrug\', \'miRNA\')_importance_for_test_{edge_type}.csv')

    LncMi_contributions_bool = LncMi_contributions.applymap(convert_to_binary)
    DrugDrug_contributions_bool = DrugDrug_contributions.applymap(convert_to_binary)
    MiDrug_contributions_bool = MiDrug_contributions.applymap(convert_to_binary)
    MiMi_contributions_bool = MiMi_contributions.applymap(convert_to_binary)
    LncDrug_contributions_bool = LncDrug_contributions.applymap(convert_to_binary)
    LncLnc_contributions_bool = LncLnc_contributions.applymap(convert_to_binary)
    rev_LncMi_contributions_bool = rev_LncMi_contributions.applymap(convert_to_binary)
    rev_LncDrug_contributions_bool = rev_LncDrug_contributions.applymap(convert_to_binary)
    rev_MiDrug_contributions_bool = rev_MiDrug_contributions.applymap(convert_to_binary)

    LncMi_avg_contributions = np.mean(LncMi_contributions_bool.mean(axis=0))
    DrugDrug_avg_contributions = np.mean(DrugDrug_contributions_bool.mean(axis=0))
    MiDrug_avg_contributions = np.mean(MiDrug_contributions_bool.mean(axis=0))
    MiMi_avg_contributions = np.mean(MiMi_contributions_bool.mean(axis=0))
    LncDrug_avg_contributions = np.mean(LncDrug_contributions_bool.mean(axis=0))
    LncLnc_avg_contributions = np.mean(LncLnc_contributions_bool.mean(axis=0))
    rev_LncMi_avg_contributions = np.mean(rev_LncMi_contributions_bool.mean(axis=0))
    rev_LncDrug_avg_contributions = np.mean(rev_LncDrug_contributions_bool.mean(axis=0))
    rev_MiDrug_avg_contributions = np.mean(rev_MiDrug_contributions_bool.mean(axis=0))

    LncMi_avg_contributions_new = (LncMi_avg_contributions + rev_LncMi_avg_contributions)/2.0
    LncDrug_avg_contributions_new = (LncDrug_avg_contributions + rev_LncDrug_avg_contributions) / 2.0
    MiDrug_avg_contributions_new = (MiDrug_avg_contributions + rev_MiDrug_avg_contributions) / 2.0
    # Calculate the total contribution
    total_contribution = LncMi_avg_contributions_new + DrugDrug_avg_contributions + MiDrug_avg_contributions_new\
                     +MiMi_avg_contributions + LncDrug_avg_contributions_new + LncLnc_avg_contributions
    # Calculate the contribution ratio of each edge type
    LncMi_proportion = LncMi_avg_contributions_new / total_contribution
    DrugDrug_proportion = DrugDrug_avg_contributions / total_contribution
    MiDrug_proportion = MiDrug_avg_contributions_new / total_contribution
    MiMi_proportion = MiMi_avg_contributions / total_contribution
    LncDrug_proportion = LncDrug_avg_contributions_new / total_contribution
    LncLnc_proportion = LncLnc_avg_contributions / total_contribution

    print('Edge importance for test ', edge_type)
    print('LncMi_proportion:',LncMi_proportion)
    print('DrugDrug_proportion:', DrugDrug_proportion)
    print('MiDrug_proportion:', MiDrug_proportion)
    print('MiMi_proportion:',MiMi_proportion)
    print('LncDrug_proportion:', LncDrug_proportion)
    print('LncLnc_proportion:', LncLnc_proportion)


# Extract the top k important nodes and edges for specific testing/validation edges, and form a subgraph
def extract_explain_node(edge_type,file_type):
    if file_type == 'test':
        path = './5.graph_important/'
    else:
        path = './5.graph_important/val_'
    drug_contributions = pd.read_csv(f'{path}drug_importance_for_test_{edge_type}.csv')
    lncRNA_contributions = pd.read_csv(f'{path}lncRNA_importance_for_test_{edge_type}.csv')
    miRNA_contributions = pd.read_csv(f'{path}miRNA_importance_for_test_{edge_type}.csv')

# Extract subgraphs (including top importance nodes and edges) that encourage specific testing/validation associations to make correct predictions
def extract_subgraph_node(edge_type,file_type,target_index,top_k):
    path_save = './5.graph_important/target_pair_explain/'
    if file_type == 'test':
        path = './5.graph_important/'
    else:
        path = './5.graph_important/val_'

    lncRNA_contributions = pd.read_csv(f'{path}lncRNA_importance_for_test_{edge_type}.csv')
    miRNA_contributions = pd.read_csv(f'{path}miRNA_importance_for_test_{edge_type}.csv')
    drug_contributions = pd.read_csv(f'{path}drug_importance_for_test_{edge_type}.csv')
    lncRNA_size = lncRNA_contributions.shape[0]
    miRNA_size =miRNA_contributions.shape[0]
    drug_size =drug_contributions.shape[0]
    # The order of node splicing is: lncRNA, miRNA, drug
    node_contributions = pd.concat([lncRNA_contributions, miRNA_contributions, drug_contributions], axis=0,ignore_index=True)
    node_for_target_pair = node_contributions[target_index]
    node_for_sorted_pair = node_for_target_pair.sort_values(ascending=False)
    top_important_node = node_for_sorted_pair[0:top_k]
    top_important_node_1 = top_important_node.reset_index()

    top_important_node_1['node_type'] = np.where(
        top_important_node_1['index'] < lncRNA_size,
        'lncRNA',
        np.where(
            np.logical_and(top_important_node_1['index'] >= lncRNA_size,
                           top_important_node_1['index'] < (lncRNA_size + miRNA_size)),
            'miRNA',
            'drug'
        )
    )
    top_important_node_1['index']  = top_important_node_1['index'].map(lambda x: modify_node_index(x, lncRNA_size, miRNA_size, drug_size))
    top_important_node_1.to_csv(f'{path_save}top_{top_k}_node_for_{file_type}_{edge_type}_{target_index}.csv', index=False)

def extract_subgraph_edge(edge_type, file_type, target_index, top_k,edge_index_all):
    path_save = './5.graph_important/target_pair_explain/'
    if file_type == 'test':
        path = './5.graph_important/'
        data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
    else:
        path = './5.graph_important/val_'
        data = torch.load('./3.heter_data/' + edge_type + '_val_data.pth')

    LncMi_contributions = pd.read_csv(f'{path}(\'lncRNA\', \'LncMi\', \'miRNA\')_importance_for_test_{edge_type}.csv')
    DrugDrug_contributions = pd.read_csv(f'{path}(\'drug\', \'DrugDrug\', \'drug\')_importance_for_test_{edge_type}.csv')
    MiDrug_contributions = pd.read_csv(f'{path}(\'miRNA\', \'MiDrug\', \'drug\')_importance_for_test_{edge_type}.csv')
    MiMi_contributions = pd.read_csv(f'{path}(\'miRNA\', \'MiMi\', \'miRNA\')_importance_for_test_{edge_type}.csv')
    LncDrug_contributions = pd.read_csv(f'{path}(\'lncRNA\', \'LncDrug\', \'drug\')_importance_for_test_{edge_type}.csv')
    LncLnc_contributions = pd.read_csv(f'{path}(\'lncRNA\', \'LncLnc\', \'lncRNA\')_importance_for_test_{edge_type}.csv')
    rev_LncMi_contributions = pd.read_csv(f'{path}(\'miRNA\', \'rev_LncMi\', \'lncRNA\')_importance_for_test_{edge_type}.csv')
    rev_LncDrug_contributions = pd.read_csv(f'{path}(\'drug\', \'rev_LncDrug\', \'lncRNA\')_importance_for_test_{edge_type}.csv')
    rev_MiDrug_contributions = pd.read_csv(f'{path}(\'drug\', \'rev_MiDrug\', \'miRNA\')_importance_for_test_{edge_type}.csv')

    LncMi_contributions_new = (LncMi_contributions + rev_LncMi_contributions)/2.0
    LncDrug_contributions_new = (LncDrug_contributions + rev_LncDrug_contributions)/2.0
    MiDrug_contributions_new = (MiDrug_contributions + rev_MiDrug_contributions )/2.0
    # The order of edge stitching is: LncMi, LncDrug, MiDrug, LncLnc, MiMi, DrugDrug
    edge_contributions = pd.concat([LncMi_contributions_new, LncDrug_contributions_new, MiDrug_contributions_new,LncLnc_contributions,MiMi_contributions,DrugDrug_contributions], axis=0,ignore_index=True)

    edge_for_target_pair = edge_contributions[target_index]
    edge_for_sorted_pair = edge_for_target_pair.sort_values(ascending=False)
    top_important_edge = edge_for_sorted_pair[0:top_k]
    top_important_edge_1 = top_important_edge.reset_index()
    top_important_edge_2 = edge_index_all.loc[top_important_edge_1['index']]
    top_important_edge_2 = top_important_edge_2.reset_index(drop=True)
    print(top_important_edge_2.index)
    print(top_important_edge_1[target_index].index)
    top_important_edge_3 =  pd.concat([top_important_edge_2, top_important_edge_1[target_index]], axis=1)
    top_important_edge_3.to_csv(f'{path_save}top_{top_k}_edge_for_{file_type}_{edge_type}_{target_index}.csv', index=False)

def construc_cytoscape_file(edge_type, file_type, target_index, top_k):
    path_save = './5.graph_important/target_pair_explain/'
    lncRNA = pd.read_csv(f'{path_save}lncRNA.csv')
    miRNA = pd.read_csv(f'{path_save}miRNA.csv')
    drug = pd.read_csv(f'{path_save}Drug.csv')
    lncRNA_num = lncRNA.shape[0]
    miRNA_num = miRNA.shape[0]
    file_path = f'{path_save}top_{top_k}_edge_for_{file_type}_{edge_type}_{target_index}.csv'
    df = pd.read_csv(file_path)
    df1 = df[['index_0', 'node_0']]
    df2 = df[['index_1', 'node_1']]
    df1.columns = ['index', 'node']
    df2.columns = ['index', 'node']
    df_concat = pd.concat([df1, df2], ignore_index=True)
    df_unique = df_concat.drop_duplicates()
    extracted_rows_name = []
    extracted_new_index = []
    # It is necessary to process the index numbers of nodes of different types and prepare files for plotting on Cytoscape, with the index sequence number of lncRNA miRNA drug
    for index, row in df_unique.iterrows():
        node_type = row['node']
        row_index = int(row['index'])

        if node_type == 'lncRNA':
            extracted_row_name = lncRNA.loc[row_index, 'x']
            new_index = row_index
        elif node_type == 'miRNA':
            extracted_row_name = miRNA.loc[row_index, 'x']
            new_index = row_index + lncRNA_num
        elif node_type == 'drug':
            extracted_row_name = drug.loc[row_index, 'Drug_Name']
            new_index = row_index + lncRNA_num + miRNA_num

        extracted_rows_name.append(extracted_row_name)
        extracted_new_index.append(new_index)

    df_unique['node_name'] = extracted_rows_name
    df_unique['new_index'] = extracted_new_index
    df_unique.columns = ['index','type','name','new_index']


    row_index_0_list = []
    row_index_1_list = []
    edge_type_list = []
    for index,row in df.iterrows():
        node_type_0 = row['node_0']
        row_index_0 = int(row['index_0'])
        node_type_1 = row['node_1']
        row_index_1 = int(row['index_1'])
        edge_type_each = node_type_0 + '-' + node_type_1

        if node_type_0 == 'lncRNA':
            row_index_0 = row_index_0
        elif node_type_0 == 'miRNA':
            row_index_0 = row_index_0 + lncRNA_num
        elif node_type_0 == 'drug':
            row_index_0 = row_index_0 + lncRNA_num + miRNA_num

        if node_type_1 == 'lncRNA':
            row_index_1 = row_index_1
        elif node_type_1 == 'miRNA':
            row_index_1 = row_index_1 + lncRNA_num
        elif node_type_1 == 'drug':
            row_index_1 = row_index_1 + lncRNA_num + miRNA_num

        row_index_0_list.append(row_index_0)
        row_index_1_list.append(row_index_1)
        edge_type_list.append(edge_type_each)

    df['new_index_0'] = row_index_0_list
    df['new_index_1'] = row_index_1_list
    df['edge_type'] = edge_type_list
    df_new = df.drop_duplicates()
    df_unique.to_csv(f'{path_save}cyto_top_{top_k}_node_for_{file_type}_{edge_type}_{target_index}.csv',index=False)
    df_new.to_csv(f'{path_save}cyto_top_{top_k}_edge_for_{file_type}_{edge_type}_{target_index}.csv', index=False)

def explain_graph_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--task_proportion", action="store_true",help="if need compute import proportion")
    parser.add_argument("-t", "--task_extract_subgraph", action="store_true",help="if need extract subgraph top edge/node for target positive association")
    parser.add_argument("-g", "--task_plot_cytoscape", action="store_true", help="if need construct network graph for using cytoscape")
    parser.add_argument("-e", "--edge_type", type=str, default="LncDrug",help="input the edge type")
    parser.add_argument("-f", "--file_type", type=str, default="test",help="input the target file type")
    parser.add_argument("-i", "--target_index",type=str, default="2", help="input the target edge index")
    parser.add_argument("-k", "--top_k", type=int, default=100, help="input the top important nodes or edges for saving")
    args = parser.parse_args()

    # Explain the contribution of different nodes/edges to predicting the overall test/validation samples for MiDrug/LncDrug tasks (calculate the overall proportion)
    edge_type = args.edge_type
    file_type = args.file_type
    target_index = args.target_index
    top_k = args.top_k
    if args.task_proportion:
       explain_node(edge_type,file_type)
       explain_edge(edge_type,file_type)

    ##%%%%%%%%%%
    # Search for important subgraphs predicted by specific testing/validation associations
    if args.task_extract_subgraph:
       edge_index_all = save_edge_index_all(edge_type, file_type)
       extract_subgraph_node(edge_type, file_type, target_index, top_k)
       extract_subgraph_edge(edge_type, file_type, target_index, top_k,edge_index_all)

    # If the top k important nodes/edges of the target test/validation edges have already been saved, convert the above files to the files required for Cytoscape drawing
    if args.task_plot_cytoscape:
        construc_cytoscape_file(edge_type, file_type, target_index, top_k)


# Example running statement:
# '- f' indicates whether to explain test_data or val_data
# '- i' indicates the str (index number) of positive sample edges in the file to be explained
# '- e' indicates whether to explain LncDrug or MiDrug tasks
# '-t' indicates if need extract subgraph top edge/node for target positive association
# '-p' indicates if need compute import proportion
# '-g' The subgraphs contributed by the top k have been extracted and the results saved before they can be modified to generate subgraph file formats that can be drawn in Cytoscape

# Only for a single testing/validation edge, identify its top important nodes and edges
# python graph_explain.py -e 'LncDrug' -f 'test' -i '3' -k 500 -t

# Calculate importance ratio of all nodes and edges, and for a single test/validation edge, find the top important nodes and edges
# python graph_explain.py -e 'LncDrug' -f 'test' -i '5' -k 300 -p

# For a single test/validation edge, identify its top important nodes and edges, and retain the file in the format required for drawing a network diagram in Cytoscape
# python graph_explain.py -e 'MiDrug' -f 'test' -i '190' -k 300 -t -g


if __name__ == "__main__":
    explain_graph_main()
