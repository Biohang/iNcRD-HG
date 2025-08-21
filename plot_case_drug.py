# %%
import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr,mannwhitneyu,pearsonr,kendalltau
import gseapy as gp


def plot_expression(AUC,Expr,target_durg,RNA_index,edge_type):
    AUC = AUC.flatten()
    expr = Expr.flatten()

    sensitive_cutoff = np.percentile(AUC, 25)
    resistant_cutoff  = np.percentile(AUC, 75)
    # Extract the index number of drug-resistant cells
    high_AUC_indices = np.where(AUC >= resistant_cutoff)[0]
    # Extract drug sensitive cell index numbers
    low_AUC_indices = np.where(AUC <= sensitive_cutoff)[0]

    # Segmenting expression data based on different drug response
    high_expr = expr[high_AUC_indices]
    low_expr = expr[low_AUC_indices]

    u_stat, p_value = mannwhitneyu(high_expr, low_expr, alternative='two-sided')
    print(f"Mann-Whitney U test p-value: {p_value:.6f}")
    AUC_resistant = AUC[high_AUC_indices]
    AUC_sensitive = AUC[low_AUC_indices]
    drug_auc = np.concatenate([AUC_resistant, AUC_sensitive])
    rna_expr = np.concatenate([high_expr, low_expr])
    pearson_corr, pearson_p_value = pearsonr(drug_auc, rna_expr)
    print(f"pearsonr_corr: {pearson_corr:.6f} ; pearson_p_value: {pearson_p_value:.6f}")

    df = pd.DataFrame({
        'sensitive_RNA_exp': low_expr,
        'resistant_RNA_exp': high_expr,
        'sensitive_AUC': AUC_sensitive,
        'resistant_AUC': AUC_resistant,
        'low_AUC_cell_indices': low_AUC_indices,
        'high_AUC_cell_indices': high_AUC_indices
    })
    df.to_csv(f'./7.case_drug/case_figure/{edge_type}_exp_auc_{target_durg}_{RNA_index}.csv', index=False)
    return p_value,pearson_corr,pearson_p_value



def plot_case_graph(edge_type,target_drug_index,top_k):
    if (edge_type == 'LncDrug'):
        RNA_expression = pd.read_csv('./7.case_drug/data/LncExp_new.csv')
    if (edge_type == 'MiDrug'):
        RNA_expression = pd.read_csv('./7.case_drug/data/MiExp_new.csv')
    drug_cell = pd.read_csv('./7.case_drug/data/drug_cell_new.csv')
    pre_results = pd.read_csv(f'./7.case_drug/results/our_new_{edge_type}_{target_drug_index}_score.csv')
    drug_cell_indices = pd.read_csv('./7.case_drug/data/drug_cell_indices.csv')
    target_drug = int(target_drug_index)
    drug_index_new = drug_cell_indices[drug_cell_indices.iloc[:, 0] == target_drug].index.tolist()[0]

    AUC = np.array(drug_cell.iloc[:, drug_index_new+1].fillna(drug_cell.iloc[:, drug_index_new+1].mean())).reshape(-1, 1)
    p_value_list = []
    pearson_corr_list = []
    pearson_p_value_list = []
    for i in range(top_k):
        RNA_index = pre_results.iloc[i, 0]
        Expr = np.array(RNA_expression.iloc[:, RNA_index+1]).reshape(-1, 1)
        p_value,pearson_corr,pearson_p_value = plot_expression(AUC, Expr,RNA_index,target_drug,edge_type)
        p_value_list.append(p_value)
        pearson_corr_list.append(pearson_corr)
        pearson_p_value_list.append(pearson_p_value)
    pd.DataFrame(p_value_list).to_csv(f'./7.case_drug/case_figure/{edge_type}_{target_drug_index}_p_value_list.csv', index=False)
    pd.DataFrame(pearson_corr_list).to_csv(f'./7.case_drug/case_figure/{edge_type}_{target_drug_index}_pearson_corr_list.csv',index=False)
    pd.DataFrame(pearson_p_value_list).to_csv(f'./7.case_drug/case_figure/{edge_type}_{target_drug_index}_pearson_p_value_list.csv',index=False)


def RNA_name_list(target_drug_index):
    pre_result_lncRNA = pd.read_csv(f'./7.case_drug/results/our_new_LncDrug_{target_drug_index}_score.csv')
    pre_result_miRNA = pd.read_csv(f'./7.case_drug/results/our_new_MiDrug_{target_drug_index}_score.csv')
    lncRNA = pd.read_csv(f'./7.case_drug/data/lncRNA.csv')
    miRNA = pd.read_csv(f'./7.case_drug/data/miRNA.csv')
    index_to_lncRNA = lncRNA['x'].to_dict()
    pre_result_lncRNA['source_lncRNA'] = pre_result_lncRNA['source'].map(index_to_lncRNA)
    index_to_miRNA = miRNA['x'].to_dict()
    pre_result_miRNA['source_miRNA'] = pre_result_miRNA['source'].map(index_to_miRNA)
    columns_order = ['source', 'source_lncRNA', 'target', 'y_scores', 'y_true']
    pre_result_lncRNA = pre_result_lncRNA.reindex(columns=columns_order)
    columns_order = ['source', 'source_miRNA', 'target', 'y_scores', 'y_true']
    pre_result_miRNA = pre_result_miRNA.reindex(columns=columns_order)
    pre_result_lncRNA.to_csv(f'./7.case_drug/results/LncDrug_{target_drug_index}_score.csv', index=False)
    pre_result_miRNA.to_csv(f'./7.case_drug/results/MiDrug_{target_drug_index}_score.csv', index=False)

def extract_gene_list(lncRNA_top,miRNA_top,target_drug_index):
    ENCORI_lncRNA = pd.read_csv('./7.case_drug/data/ENCORI_hg38_lncRNA-RNA_all.txt', skiprows=3, sep='\t')
    ENCORI_miRNA = pd.read_csv('./7.case_drug/data/ENCORI_hg38_miRNA-mRNA_all.txt', skiprows=3, sep='\t')
    # Interactions with interaction_num greater than 10 are considered to have high confidence
    # Interactions with a TDMDScore greater than 0.8 are considered to have high TDMD potential
    interaction_num_thresholds =  10
    TDMDScore_thresholds = 0.8
    ENCORI_lncRNA_subset = ENCORI_lncRNA[
        (ENCORI_lncRNA['geneType'] == 'lncRNA') &
        (ENCORI_lncRNA['pairGeneType'] == 'protein_coding') &
        (ENCORI_lncRNA['interactionNum'] > interaction_num_thresholds) &
        (ENCORI_lncRNA['geneName'].isin(lncRNA_top))
    ]
    lncRNA_gene_list = list(set(ENCORI_lncRNA_subset['pairGeneName'].tolist()))
    ENCORI_miRNA_subset = ENCORI_miRNA[
        (ENCORI_miRNA['geneType'] == 'protein_coding') &
        (ENCORI_miRNA['TDMDScore'] > TDMDScore_thresholds) &
        (ENCORI_miRNA['miRNAname'].str.upper().isin(miRNA_top))
    ]
    miRNA_gene_list = list(set(ENCORI_miRNA_subset['geneName'].tolist()))
    gene_list = list(set(lncRNA_gene_list + miRNA_gene_list))
    lncRNA_gene_df = pd.DataFrame({"gene": lncRNA_gene_list})
    miRNA_gene_df = pd.DataFrame({"gene": miRNA_gene_list})
    gene_df = pd.DataFrame({"gene": gene_list})
    lncRNA_gene_df.to_csv(f'./7.case_drug/results/lncRNA_gene_df_{target_drug_index}.txt', index=False, header=False)
    miRNA_gene_df.to_csv(f'./7.case_drug/results/miRNA_gene_df_{target_drug_index}.txt', index=False, header=False)
    gene_df.to_csv(f'./7.case_drug/results/gene_df_{target_drug_index}.txt', index=False, header=False)

    return lncRNA_gene_list,miRNA_gene_list,gene_list

def gene_set_graph(target_drug_index,top_k):
    pre_result_lncRNA = pd.read_csv(f'./7.case_drug/results/LncDrug_{target_drug_index}_score.csv')
    pre_result_miRNA = pd.read_csv(f'./7.case_drug/results/MiDrug_{target_drug_index}_score.csv')
    lncRNA_top = pre_result_lncRNA.iloc[:top_k, 1].tolist()
    miRNA_top = pre_result_miRNA.iloc[:top_k, 1].tolist()

    #LncRNA and miRNA cannot be directly enriched for analysis, therefore lncRNA-mRNA and miRNA-target interaction were downloaded from the starBase (ENCORI) 
    #curl 'https://rnasysu.com/encori/api/RNARNA/?assembly=hg38&geneType=lncRNA&RNA=all&interNum=2&expNum=2&cellType=all' > ENCORI_hg38_lncRNA-RNA_all.txt
    #curl 'https://rnasysu.com/encori/api/miRNATarget/?assembly=hg38&geneType=mRNA&miRNA=all&clipExpNum=5&degraExpNum=1&pancancerNum=10&programNum=3&program=PITA,miRanda&target=all&cellType=all' > ENCORI_hg38_miRNA-mRNA_all.txt
    #Extract top lnRNA and top miRNA related mRNA from the above data, and then perform gene enrichment analysis/pathway analysis
    lncRNA_gene_list,miRNA_gene_list,gene_list = extract_gene_list(lncRNA_top,miRNA_top,target_drug_index)

    go_enrichment_1 = gp.enrichr(
        gene_list=f"./7.case_drug/results/gene_df_{target_drug_index}.txt", 
        gene_sets="GO_Biological_Process_2025",
        organism="human",
        outdir="./7.case_drug/results/go_enrichment_results", 
        cutoff=0.05
    )
    go_enrichment_2 = gp.enrichr(
        gene_list=f"./7.case_drug/results/gene_df_{target_drug_index}.txt", 
        gene_sets="GO_Cellular_Component_2025",
        organism="human",
        outdir="./7.case_drug/results/go_enrichment_results", 
        cutoff=0.05
    )
    go_enrichment_3 = gp.enrichr(
        gene_list=f"./7.case_drug/results/gene_df_{target_drug_index}.txt",
        gene_sets="GO_Molecular_Function_2025",
        organism="human",
        outdir="./7.case_drug/results/go_enrichment_results",
        cutoff=0.05
    )
    KEGG_enrichment = gp.enrichr(
        gene_list=f"./7.case_drug/results/gene_df_{target_drug_index}.txt", 
        gene_sets="KEGG_2021_Human", 
        organism="human",
        outdir="./7.case_drug/results/go_enrichment_results",
        cutoff=0.05 
    )
    go_enrichment_1.res2d.to_csv(f"7.case_drug/results/go_enrichment_results/GO_BP_{target_drug_index}.csv", index=False)
    go_enrichment_2.res2d.to_csv(f"7.case_drug/results/go_enrichment_results/GO_CC_{target_drug_index}.csv", index=False)
    go_enrichment_3.res2d.to_csv(f"7.case_drug/results/go_enrichment_results/GO_MF_{target_drug_index}.csv", index=False)
    KEGG_enrichment.res2d.to_csv(f"7.case_drug/results/go_enrichment_results/KEGG_{target_drug_index}.csv",index=False)
    # Generate enriched bubble map
    gp.dotplot(
        df=go_enrichment_1.res2d, 
        color="P-value", 
        y="Term",
        x="Combined Score", 
        size=8,
        top_term=10,
        figsize=(4, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/GO_BP_{target_drug_index}.jpg"
    )
    gp.dotplot(
        df=go_enrichment_2.res2d, 
        color="P-value", 
        y="Term",
        x="Combined Score", 
        size=8,
        cmap="plasma_r",
        top_term=10,
        figsize=(4, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/GO_CC_{target_drug_index}.jpg"
    )
    gp.dotplot(
        df=go_enrichment_3.res2d,
        color="P-value", 
        y="Term",
        x="Combined Score", 
        size=8,
        cmap="magma_r",
        top_term=10,
        figsize=(4, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/GO_MF_{target_drug_index}.jpg"
    )

    gp.dotplot(
        df=KEGG_enrichment.res2d, 
        color="P-value", 
        y="Term",
        x="Combined Score",
        cmap="cividis_r",
        size=8,
        top_term=10,
        figsize=(4, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/KEGG_{target_drug_index}.jpg"
    )
    # Generate enrichment bar chart
    colors = plt.cm.plasma(go_enrichment_2.res2d['P-value'].rank(pct=True))
    gp.barplot(
        df=go_enrichment_1.res2d,
        column="P-value",
        x="Term",
        y="Combined Score",
        color="#669aba",
        top_term=10,
        figsize=(9, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/GO_BP_bar_{target_drug_index}.jpg"
    )
    gp.barplot(
        df=go_enrichment_2.res2d,
        column="P-value",
        x="Term",
        y="Combined Score",
        color="#8ab07c",
        top_term=10,
        figsize=(9, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/GO_CC_bar_{target_drug_index}.jpg"
    )
    gp.barplot(
        df=go_enrichment_3.res2d,
        column="P-value",
        color="#e7c66b",
        x="Term",
        y="Combined Score",
        size=8,
        top_term=10,
        figsize=(9, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/GO_MF_bar_{target_drug_index}.jpg"
    )
    gp.barplot(
        df=KEGG_enrichment.res2d,
        column="P-value",
        color="#e66d50",
        x="Term",
        y="Combined Score",
        size=8,
        top_term=10,
        figsize=(9, 6),
        fontsize=10,
        ofname=f"./7.case_drug/results/go_enrichment_results/KEGG_bar_{target_drug_index}.jpg"
    )

#In the benchmark dataset, there are 9 drugs that have more than 30 associations with resistance to two types of RNA, with index numbers of 0, 1, 2, 3, 4, 5, 6, 10, and 13. 
# Example running statement:
# python plot_case_drug.py -i '0' -k 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--target_drug_index",type=str, default="0", help="input the target edge index")
    parser.add_argument("-k", "--top_k", type=int, default=200, help="input the top important nodes or edges for saving")
    args = parser.parse_args()
    target_drug_index = args.target_drug_index
    top_k = args.top_k

    plot_case_graph('LncDrug', target_drug_index, top_k)
    plot_case_graph('MiDrug', target_drug_index, top_k)

    #The following is the section on drawing gene enrichment results. Run the following statement and take top_k=200
    #RNA_name_list(target_drug_index)
    #gene_set_graph(target_drug_index,top_k)