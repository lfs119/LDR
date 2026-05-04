# utils/visualize_tree.py

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def visualize_tree_structure(frag_indices, ecfps, bond_types, bondMapNums, 
                           save_path=None, title="Fragment Tree Structure Analysis"):
    """
    绘制 fragment tree 的 depth 和 degree 分布直方图
    """
    from tree import make_tree

    # 构建树
    tree = make_tree(frag_indices, ecfps, bond_types, bondMapNums)
    tree.set_all_positional_encoding(d_pos=64)  
    g = tree.dgl_graph

    # 提取数据
    depths = g.ndata['depth'].cpu().numpy()
    degrees = g.ndata['degree'].cpu().numpy()
    fids = g.ndata['fid'].squeeze(-1).cpu().numpy()

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Depth 分布
    sns.histplot(depths, ax=axes[0], bins=np.arange(depths.max()+2)-0.5, kde=True, color='skyblue')
    axes[0].set_title('Node Depth Distribution')
    axes[0].set_xlabel('Tree Depth')
    axes[0].set_ylabel('Count')

    # Degree 分布
    sns.histplot(degrees, ax=axes[1], bins=np.arange(degrees.max()+2)-0.5, kde=True, color='salmon')
    axes[1].set_title('Node Degree (Children) Distribution')
    axes[1].set_xlabel('Number of Children')
    axes[1].set_ylabel('Count')

    # 结构信息表格（前10个节点）
    table_data = [[f"frag_{fid}", d, deg] for fid, d, deg in zip(fids, depths, degrees)]
    table = axes[2].table(cellText=table_data[:10],
                          colLabels=['Fragment', 'Depth', 'Degree'],
                          loc='center',
                          cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    axes[2].axis('off')
    axes[2].set_title('First 10 Nodes Info')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()



frag_indices = [0, 1, 2, 3]
ecfps = torch.randn(4, 256)
bond_types = [1, 2, 1]
bondMapNums = [[1], [1, 2], [2], []]

visualize_tree_structure(frag_indices, ecfps, bond_types, bondMapNums,
                       save_path="tree_analysis.png")