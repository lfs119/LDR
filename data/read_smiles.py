from rdkit import Chem

# # 读取 PDB 文件
# mol = Chem.MolFromPDBFile("7KAC_W9D.pdb", sanitize=False)  # sanitize=False 避免自动预处理

# # 转换为 SMILES
# smiles = Chem.MolToSmiles(mol, isomericSmiles=True)  # 保留立体化学信息
# print("7KAC_W9D.pdb to SMILES")
# print(smiles)

# mol = Chem.MolFromPDBFile("7NVH_AVO.pdb")  # sanitize=False 避免自动预处理

# # 转换为 SMILES
# smiles = Chem.MolToSmiles(mol, isomericSmiles=True)  # 保留立体化学信息
# print("7NVH_AVO.pdb to SMILES")
# print(smiles)

# mol = Chem.MolFromPDBFile("8TXZ_A1N.pdb", sanitize=True)  # sanitize=False 避免自动预处理

# # 转换为 SMILES
# smiles = Chem.MolToSmiles(mol, isomericSmiles=True)  # 保留立体化学信息
# print("8TXZ_A1N.pdb to SMILES")
# print(smiles)

# mol = Chem.SDMolSupplier("/raid/home/xukai/FRATTVAE/data/A1N_ideal.sdf")
# smiles = Chem.MolToSmiles(mol[0], kekuleSmiles=True)  
# print(smiles)

mol = Chem.SDMolSupplier("/raid/home/xukai/FRATTVAE/data/tps_flow/7C2N/7C2N_FGO.sdf")
smiles = Chem.MolToSmiles(mol[0], kekuleSmiles=True)  
print(smiles)

# mol = Chem.MolFromPDBFile("9EO4_coc.pdb", sanitize=True)  # sanitize=False 避免自动预处理

# # 转换为 SMILES
# smiles = Chem.MolToSmiles(mol, isomericSmiles=True)  # 保留立体化学信息
# print("9EO4_coc.pdb to SMILES")
# print(smiles)

# 7KAC_W9D.pdb to SMILES
# CC1(C)O[C@@H](O)C2CCC(NC3NCC(C4NNCO4)C(N[C@H](CO)C4CCCCC4)N3)CC21
# 8TXZ_A1N.pdb to SMILES
# C[C@@H]1CN(C2CC(C34NN3C3CCC(OC5(C)CC5)CC34)NCN2)C[C@H](C)O1
# 9EO4_coc.pdb to SMILES
# COC(O)[C@H]1[C@@H](OC(O)C2CCCCC2)C[C@@H]2CC[C@H]1N2C

# aid2282_clean.py
# import pandas as pd
# from rdkit import Chem

# # 1. 读文件
# df = pd.read_csv('AID_2283_datatable.csv')

# # 2. 基础过滤
# df = df[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active'].copy()
# # df = df.dropna(subset=['PUBCHEM_EXT_DATASOURCE_SMILES', 'Percentage'])

# # # 3. 抑制率 ➜ pIC50（近似）
# # def pct2pic50(p):
# #     p = float(p)
# #     if p >= 50:   return 5.0
# #     if p >= 20:   return 4.3 + (p - 20) / 30 * 0.7
# #     return None   # < 20 % 丢弃

# # df['pIC50'] = df['Percentage'].apply(pct2pic50)
# # df = df.dropna(subset=['pIC50'])

# # 4. 去盐 & 标准化 SMILES
# def canon_smiles(s):
#     try:
#         return Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
#     except:
#         return None

# df['smiles'] = df['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canon_smiles)
# df = df.dropna(subset=['smiles']).drop_duplicates('smiles')
# out = df['smiles']
# # 5. 最终输出
# # out = df[['smiles', 'pIC50']].sort_values('pIC50', ascending=False)
# out.to_csv('hpk1_clean.csv', index=False)
# print('✅ 清洗完成！行数：', out.shape[0])


