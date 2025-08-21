import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np

# Read CSV file
file_path = './2.construct_network/drug_smiles.csv' 
df = pd.read_csv(file_path)

# Extract CID and SMILES
cids = df['drug_cid'].tolist()
smiles_list = df['smiles'].tolist()

# Convert SMILES into molecular objects
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

# Generate molecular fingerprints (using Morgan fingerprints here)
fps = [AllChem.GetMorganFingerprint(mol, 2) for mol in mols]

# Initialize similarity matrix
num_drugs = len(fps)
similarity_matrix = np.zeros((num_drugs, num_drugs))

# Calculate Tanimoto similarity between drug pairs
for i in range(num_drugs):
    for j in range(i, num_drugs):
        similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity


np.fill_diagonal(similarity_matrix, 1)
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df.to_csv('./2.construct_network/drugSimMat.csv',index=False, header=False)