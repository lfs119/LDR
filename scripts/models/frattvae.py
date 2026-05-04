import numpy as np
from joblib import Parallel, delayed

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv

from utils.tree import FragmentTree, get_pad_features
from utils.mask import generate_square_subsequent_mask
from utils.construct import constructMol, constructMolwithTimeout


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, *args, **kwargs):
        return x + self.g * self.fn(x, *args, **kwargs)

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, d_model)
        self.beta = nn.Linear(cond_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x, cond):
        x_norm = self.layer_norm(x)
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return gamma * x_norm + beta

class TreeAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, prop_dim=0, use_gnn=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_cond = prop_dim > 0

        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-Forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # GNN for tree structure
        if use_gnn:
            self.gnn = GATConv(d_model, d_model, heads=1, dropout=dropout, concat=False)
            self.gnn_norm = nn.LayerNorm(d_model)
            # self.gnn_rezero = nn.Parameter(torch.ones(1) * 0.1)
            self.gnn_rezero = 1

        # AdaLN for property condition
        if self.use_cond:
            self.adaln1 = AdaptiveLayerNorm(d_model, prop_dim)
            self.adaln2 = AdaptiveLayerNorm(d_model, prop_dim)

        self.dropout = nn.Dropout(dropout)
        # self.attn_rezero = nn.Parameter(torch.ones(1) * 0.1)
        # self.ffn_rezero = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, prop_cond=None, edge_index=None,
                src_mask=None, src_key_padding_mask=None):  
        residual = x
        
        if self.use_cond and prop_cond is not None:
            x = self.adaln1(x, prop_cond)

        x_norm = self.norm1(x)
        attn_out = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        # x = residual + self.attn_rezero * self.dropout(attn_out)
        x = residual + 1 * self.dropout(attn_out)

        # GNN Message Passing on Fragment Tree
        if self.use_gnn and edge_index is not None and x.size(0) > 1:
            gnn_out = self.gnn(x, edge_index)
            x = x + self.gnn_rezero * self.dropout(gnn_out)
            x = self.gnn_norm(x)

        # FFN
        residual = x
        
        if self.use_cond and prop_cond is not None:
            x = self.adaln2(x, prop_cond)
        
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        # x = residual + self.ffn_rezero * self.dropout(ffn_out)
        x = residual + 1 * self.dropout(ffn_out)

        return x

class TreeAttentionDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 d_ff, 
                 dropout=0.1, 
                 prop_dim=0, 
                 use_gnn=True,):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_cond = prop_dim > 0

        # ========== Self-Attention ==========
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # ========== Cross-Attention ==========
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_cross = nn.LayerNorm(d_model)  
        # ========== Feed-Forward ==========
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # ========== GNN for Tree Structure ==========
        if use_gnn:
            # 使用 GATv2Conv 支持 edge_attr
            self.gnn = GATConv(
                in_channels=d_model,
                out_channels=d_model,
                heads=1,
                concat=False,
                dropout=dropout
            )
            self.gnn_norm = nn.LayerNorm(d_model)
            # self.gnn_rezero = nn.Parameter(torch.ones(1) * 0.1)
            self.gnn_rezero = 1
            

        # ========== AdaLN for Property Condition ==========
        if self.use_cond:
            self.adaln1 = AdaptiveLayerNorm(d_model, prop_dim)
            self.adaln2 = AdaptiveLayerNorm(d_model, prop_dim)

        # ========== Dropout & Rezero ==========
        self.dropout = nn.Dropout(dropout)
        # self.attn_rezero = nn.Parameter(torch.ones(1) * 0.1)  # self-attn
        # self.cross_rezero = nn.Parameter(torch.ones(1) * 0.1)  # cross-attn
        # self.ffn_rezero = nn.Parameter(torch.ones(1) * 0.1)

        self.attn_rezero = 1  # self-attn
        self.cross_rezero = 1 # cross-attn
        self.ffn_rezero = 1

    def forward(self,
                x,                      # [B, T, D]  当前解码状态
                memory,                 # [B, S, D]  编码器输出
                prop_cond=None,         # [B, P]     属性条件
                edge_index=None,        # [2, E]     树结构边
                tgt_mask=None,          # [T, T]     目标掩码（因果）
                memory_mask=None,       # [T, S]     可选
                tgt_key_padding_mask=None,      # [B, T]   目标 padding 掩码
                memory_key_padding_mask=None,   # [B, S]   记忆 padding 掩码
                ):
        """
        x: 当前解码器输入（已嵌入）
        memory: 编码器输出
        """
        residual = x

        # --- AdaLN 调制（可选）---
        if self.use_cond and prop_cond is not None:
            x = self.adaln1(x, prop_cond)
        
        x_norm = self.norm1(x)
        # --- Self-Attention ---
        self_attn_out = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        x = residual + self.attn_rezero * self.dropout(self_attn_out)
        

        # --- Cross-Attention ---
        residual = x
        x_norm = self.norm_cross(x) 

        if self.use_cond and prop_cond is not None:
            x_norm = self.adaln1(x_norm, prop_cond)  # 可复用 adaln1

        cross_attn_out = self.cross_attn(
            x_norm, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        x = residual + self.cross_rezero * self.dropout(cross_attn_out)

       
        if self.use_gnn and edge_index is not None and x.size(1) > 1:
            gnn_out = self.gnn(x, edge_index)  # 确保维度匹配
            x = x + self.gnn_rezero * self.dropout(gnn_out)
            x = self.gnn_norm(x)

        # --- FFN ---
        residual = x
        
        if self.use_cond and prop_cond is not None:
            x = self.adaln2(x, prop_cond)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.ffn_rezero * self.dropout(ffn_out)

        return x    
# ======================
# Encoder & Decoder
# ======================

class EnhancedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1, prop_dim=0, use_gnn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TreeAttentionLayer(d_model, nhead, d_ff, dropout, prop_dim, use_gnn=use_gnn)
            for _ in range(num_layers)
        ])

    def forward(self, src, prop_cond=None, edge_index=None, mask=None, src_key_padding_mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, prop_cond=prop_cond, edge_index=edge_index, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x


class EnhancedTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout=0.1, prop_dim=0, use_gnn=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TreeAttentionDecoderLayer(d_model, nhead, d_ff, dropout, prop_dim, use_gnn=use_gnn)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, prop_cond=None, edge_index=None, tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, prop_cond=prop_cond, edge_index=edge_index, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)
        return x


class TreePositionalEncoding(nn.Module):
    """
    Reference: https://github.com/microsoft/icecaps/blob/master/icecaps/estimators/abstract_transformer_estimator.py
    """
    def __init__(self, d_model: int, d_pos: int, depth: int, width: int) -> None:
        super().__init__()
        self.d_params = d_pos // (depth * width)
        self.d_model = d_model
        self.depth = depth
        self.width = width
        self.params = nn.Parameter(torch.randn(self.d_params), requires_grad= True)
        self.fc = nn.Linear(self.d_params * self.depth * self.width, d_model)

        # initialization
        self.tree_weights = None

    def forward(self, positions: torch.Tensor):
        """
        positions: shape= (Batch_size, Length, depth * width)
        """
        if self.training | (self.tree_weights is None):
            self._update_weights()
        treeified = positions.unsqueeze(-1) * self.tree_weights.to(positions.device)     # (B, L, depth*width, d_param)
        treeified = treeified.flatten(start_dim= 2)                                      # (B, L, depth*width*d_param = d_pos)
        if treeified.shape[-1] != self.d_model:
            treeified = self.fc(treeified)

        return treeified
    
    def _update_weights(self):
        params = torch.tanh(self.params)
        tiled_tree_params = params.view(1, 1, -1).repeat(self.depth, self.width, 1)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device= params.device).view(-1, 1, 1).repeat(1, self.width, self.d_params)
        tree_norm = torch.sqrt((1 - params.square()) * self.d_model / 2)
        self.tree_weights = (tiled_tree_params ** tiled_depths) * tree_norm
        self.tree_weights = self.tree_weights.view(self.depth * self.width, self.d_params)


class EnhancedTreePositionalEncoding(nn.Module):
    """
    Reference: https://github.com/microsoft/icecaps/blob/master/icecaps/estimators/abstract_transformer_estimator.py
    """
    def __init__(self, d_model: int, d_pos: int, depth: int, width: int, d_add_pos=32) -> None:
        super().__init__()
        self.d_params = d_pos // (depth * width)
        self.d_model = d_model
        self.depth = depth
        self.width = width
        self.params = nn.Parameter(torch.randn(self.d_params), requires_grad= True)
        self.fc = nn.Linear(self.d_params * self.depth * self.width, d_model)
        self.pos_add = nn.Linear(d_pos+d_add_pos, d_pos)

        # initialization
        self.tree_weights = None

    def forward(self, positions: torch.Tensor):
        """
        positions: shape= (Batch_size, Length, depth * width)
        """
        if self.training | (self.tree_weights is None):
            self._update_weights()

        positions = self.pos_add(positions)
        treeified = positions.unsqueeze(-1) * self.tree_weights.to(positions.device)     # (B, L, depth*width, d_param)
        treeified = treeified.flatten(start_dim= 2)                                      # (B, L, depth*width*d_param = d_pos)
        if treeified.shape[-1] != self.d_model:
            treeified = self.fc(treeified)

        return treeified
    
    def _update_weights(self):
        params = torch.tanh(self.params)
        tiled_tree_params = params.view(1, 1, -1).repeat(self.depth, self.width, 1)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device= params.device).view(-1, 1, 1).repeat(1, self.width, self.d_params)
        tree_norm = torch.sqrt((1 - params.square()) * self.d_model / 2)
        self.tree_weights = (tiled_tree_params ** tiled_depths) * tree_norm
        self.tree_weights = self.tree_weights.view(self.depth * self.width, self.d_params)
        

class FRATTVAE(nn.Module):
    def __init__(self, num_tokens: int, depth: int, width: int, 
                 feat_dim: int= 2048, latent_dim: int= 256,
                 d_model: int= 512, d_ff: int= 2048, num_layers: int= 6, nhead: int= 8, 
                 activation: str= 'gelu', dropout: float= 0.1, n_jobs: int= 4) -> None:
        super().__init__()
        assert activation in ['relu', 'gelu']
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.depth = depth
        self.width = width
        self.n_jobs = n_jobs

        self.embed = nn.Embedding(num_embeddings= 1, embedding_dim= d_model)      # <root>
        self.fc_ecfp = nn.Sequential(nn.Linear(feat_dim, feat_dim//2),
                                     nn.Linear(feat_dim//2, d_model))
        self.PE = TreePositionalEncoding(d_model= d_model, d_pos= max(d_model, depth*width), depth= depth, width= width)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff,
                                                   dropout= self.dropout, activation= activation, batch_first= True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)

        # vae
        self.fc_vae = nn.Sequential(nn.Linear(d_model, latent_dim), 
                                    nn.Linear(latent_dim, 2*latent_dim))

        # transformer decoder
        self.fc_memory = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff,
                                                   dropout= self.dropout, activation= activation, batch_first= True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers= num_layers)
        self.fc_dec = nn.Linear(d_model, num_tokens)

        # for decode smiles
        self.labels = None

    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None, 
                tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None,
                frag_ecfps: torch.Tensor= None, ndummys: torch.Tensor= None, 
                max_nfrags: int= 20, free_n: bool= False, sequential: bool= None, conditions: torch.Tensor= None):
        """
        encode and decode
        Decode in parallel when training process.
        """
        sequential = not self.training if sequential is None else sequential
        
        # (2048,256)
        z, mu, ln_var = self.encode(features, positions, src_mask, src_pad_mask, conditions)
        if sequential:
            output, _ = self.sequential_decode(z, frag_ecfps, ndummys, conditions= conditions, max_nfrags= max_nfrags, free_n= free_n)
        else:
            output = self.decode(z, features, positions, tgt_mask, tgt_pad_mask, conditions)

        return z, mu, ln_var, output


    def encode(self, features: torch.Tensor, positions: torch.Tensor, 
               src_mask: torch.Tensor= None, src_pad_mask: torch.Tensor= None, conditions: torch.Tensor= None):
        """
        features: shape= (Batch_size, Length, feat_dim)
        positions: shape= (Batch_size, Length, depth * width)
        condtions: shape= (Batch_size, num_conditions, d_model) if conditional vae.
        src_mask: source mask for masked attention, shape= (Length+num_conditions+1, Length+num_conditions+1)
        src_pad_mask: shape = (Batch_size, Length+num_conditions+1)
        """
        num_conditions = conditions.shape[1] if conditions is not None else 0

        # positional embbeding
        src = self.fc_ecfp(features) + self.PE(positions)           # (B, L, d_model),  * math.sqrt(self.d_model)?

        # attach super root
        root_embed = self.embed(src.new_zeros(src.shape[0], 1).long())     
        if num_conditions > 0:
            src = torch.cat([conditions, root_embed, src], dim= 1)  # (B, L+num_conditions+1, d_model)
        else:
            src = torch.cat([root_embed, src], dim= 1)              # (B, L+1, d_model)

        # transformer encoding
        out = self.encoder(src, mask= src_mask, src_key_padding_mask= src_pad_mask)
        out = out[:, num_conditions, :].squeeze(1)

        # vae
        mu, ln_var = self.fc_vae(out).chunk(2, dim= -1)
        z = self.reparameterization_trick(mu, ln_var)               # (B, latent_dim)

        return z, mu, ln_var
    

    def decode(self, z: torch.Tensor, features: torch.Tensor, positions: torch.Tensor,
               tgt_mask: torch.Tensor= None, tgt_pad_mask: torch.Tensor= None, conditions: torch.Tensor= None):
        """
        z: encoder output. shape= (Batch_size, latent_dim)
        features: shape= (Batch_size, Length, feat_dim)
        positions: shape= (Batch_size, Length, depth * width)
        condtions: shape= (Batch_size, num_conditions, d_model) if conditional vae.
        tgt_mask: target mask for masked attention, shape= (Length+1, Length+1)
        tgt_pad_mask: target mask for padding, shape= (Batch_size, Length+1)

        output: logits of label preditions, shape= (Batch_size, Length+1, num_labels)
        """
        num_conditions = conditions.shape[1] if conditions is not None else 0

        # latent variable to memory
        memory = self.fc_memory(z).unsqueeze(1)                     # (B, 1, d_model)

        # postional embedding 
        tgt = self.fc_ecfp(features) + self.PE(positions)           # (B, L, d_model)

        # attach supur root
        root_embed = self.embed(tgt.new_zeros(tgt.shape[0], 1).long())    
        if num_conditions > 0:
            tgt = torch.cat([conditions, root_embed, tgt], dim= 1)  # (B, L+num_conditions+1, d_model)
        else:
            tgt = torch.cat([root_embed, tgt], dim= 1)              # (B, L+1, d_model)

        # transformer decoding
        out = self.decoder(tgt, memory, tgt_mask= tgt_mask, tgt_key_padding_mask= tgt_pad_mask)
        out = self.fc_dec(out[:, num_conditions:])                  # (B, L+1, num_tokens)

        return out
    

    def sequential_decode(self, z: torch.Tensor, frag_ecfps: torch.Tensor, ndummys: torch.Tensor, 
                          max_nfrags: int= 30, free_n: bool= False, asSmiles: bool= False, conditions: torch.Tensor= None) -> list:
        """
        z: latent variable. shape= (Batch_size, latent_dim)
        frag_ecfps: fragment ecfps. shape= (num_labels, feat_dim)
        ndummys: The degree of a fragment means how many children it has. shape= (num_labels, )
        max_nfrags: the maximum number of fragments
        free_n: if False, tree positional encoding as all nodes have n children.

        output: list of fragment tree
        """
        batch_size = z.shape[0]
        device = z.device
        num_conditions = conditions.shape[1] if conditions is not None else 0

        # latent variabel to memory
        memory = self.fc_memory(z).unsqueeze(1)

        # root prediction
        root_embed = self.embed(torch.zeros(batch_size, 1, device= device).long())
        if num_conditions > 0:
            root_embed = torch.cat([conditions, root_embed], dim= 1)    # (B, num_conditions+1, d_model)
        tgt_pad_mask = torch.all(root_embed==0, dim= -1).to(device)     # (B, num_conditions+1)
        out = self.decoder(root_embed, memory, tgt_key_padding_mask= tgt_pad_mask)
        out = self.fc_dec(out[:, num_conditions:])
        root_idxs = out.argmax(dim= -1).flatten()      # (B, )
        # out = out.argsort(dim= -1, descending= True).squeeze(1)
        # root_idxs = torch.where(out[:,0]!=0, out[:,0], out[:,1])    # (B, )
        
        continues = []
        target_ids = [0] * batch_size
        target_ids_list = [[0] for _ in range(batch_size)]
        tree_list = [FragmentTree() for _ in range(batch_size)]
        for i, idx in enumerate(root_idxs):
            parent_id = tree_list[i].add_node(parent_id= None, feature= frag_ecfps[idx], fid= idx.item(), bondtype= 0)
            assert parent_id == 0
            tree_list[i].set_positional_encoding(parent_id, d_pos= self.depth * self.width)
            continues.append(ndummys[idx].item() > 0)

        nfrags = 1
        while (nfrags < max_nfrags) & (sum(continues) > 0):
            # features
            tgt_mask = generate_square_subsequent_mask(length= nfrags+num_conditions+1).to(device)
            tgt_mask[:, :num_conditions+1] = 0      # no sequence mask of conditions
            tgt_pad_mask = torch.hstack([tgt_pad_mask, tgt_pad_mask.new_full(size= (batch_size, 1), fill_value= False)])
            features = get_pad_features(tree_list, key= 'x', max_nodes_num= nfrags).to(device)
            positions = get_pad_features(tree_list, key= 'pos', max_nodes_num= nfrags).to(device)
            assert features.shape[0] == positions.shape[0]

            # forward
            tgt = self.fc_ecfp(features) + self.PE(positions)
            tgt = torch.cat([root_embed, tgt], dim= 1)          # (B, nfrags+num_conditions+1, d_model)

            out = self.decoder(tgt, memory, tgt_mask= tgt_mask, tgt_key_padding_mask= tgt_pad_mask)
            out = self.fc_dec(out[:, num_conditions:])              # (B, nfrags+1, num_labels)
            
            new_idxs = out[:, -1, :].argmax(dim= -1).flatten()      # (B,)

            # add node
            for i, idx in enumerate(new_idxs):
                if continues[i]:
                    if ndummys[idx] == 0:   # don't generate salt compounds.
                        idx = torch.tensor(0)
                    if idx != 0:
                        parent_id = target_ids[i]
                        add_node_id = tree_list[i].add_node(parent_id= parent_id, feature= frag_ecfps[idx], fid= idx.item(), bondtype= 1)
                        parent_fid = tree_list[i].dgl_graph.ndata['fid'][parent_id].item()
                        num_sibling = ndummys[parent_fid].item() - 1 if parent_id > 0 else ndummys[parent_fid].item()
                        if free_n:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling= num_sibling, d_pos= self.depth * self.width)
                        else:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling= self.width, d_pos= self.depth * self.width)
                        level = tree_list[i].dgl_graph.ndata['level'][add_node_id].item()

                        # compare the current number of siblings with the ideal number of siblings
                        if (len(tree_list[i].dgl_graph.predecessors(parent_id)) >= num_sibling):
                            target_ids_list[i].pop(-1)

                        # whether the node has children
                        if (ndummys[idx] > 1) & (self.depth > level):
                            target_ids_list[i].append(add_node_id)

                    continues[i] = bool(target_ids_list[i]) if (idx != 0) else False
                    target_ids[i] = target_ids_list[i][-1] if continues[i] else 0
            nfrags += 1

        if asSmiles:
            if self.labels is not None:
                # outputs = [constructMol(self.labels[tree.dgl_graph.ndata['fid'].squeeze(-1).tolist()], tree.adjacency_matrix().tolist()) for tree in tree_list]
                outputs = Parallel(n_jobs= self.n_jobs)(delayed(constructMol)(self.labels[tree.dgl_graph.ndata['fid'].squeeze(-1).tolist()], tree.adjacency_matrix().tolist()) for tree in tree_list)
                # outputs = Parallel(n_jobs= self.n_jobs, backend='threading')(delayed(constructMolwithTimeout)(self.labels[tree.dgl_graph.ndata['fid'].squeeze(-1).tolist()], tree.adjacency_matrix().tolist()) for tree in tree_list)
            else:
                raise ValueError('If asSmiles= True, please set labels. exaple; self.set_labels(labels)')
        else:
            outputs = tree_list

        return outputs, None

    def reparameterization_trick(self, mu, ln_var):
        eps = torch.randn_like(mu)
        z = mu + torch.exp(ln_var / 2) * eps if self.training else mu

        return z
    
    def set_labels(self, labels):
        if type(labels) == np.ndarray:
            self.labels = labels
        else:
            self.labels = np.array(labels)

class FRATTVAE_Enhanced(FRATTVAE):
    def __init__(self, num_tokens: int, depth: int, width: int, 
                 feat_dim: int= 2048, latent_dim: int= 256,
                 d_model: int= 512, d_ff: int= 2048, num_layers: int= 6, nhead: int= 8, 
                 activation: str= 'gelu', dropout: float= 0.1, n_jobs: int= 4, prop_dim: int=0, use_gnn=False) -> None:
        super().__init__(num_tokens, depth, width, feat_dim, latent_dim, d_model, d_ff, num_layers, nhead, activation, dropout, n_jobs)
        assert activation in ['relu', 'gelu']
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.depth = depth
        self.width = width
        self.n_jobs = n_jobs

        self.embed = nn.Embedding(num_embeddings= 1, embedding_dim= d_model)      # <root>
        self.fc_ecfp = nn.Sequential(nn.Linear(feat_dim, feat_dim//2),
                                     nn.Linear(feat_dim//2, d_model))
        

        self.PE = TreePositionalEncoding(d_model= d_model, d_pos= max(d_model, depth*width), depth= depth, width= width)
        # self.PE = EnhancedTreePositionalEncoding(d_model= d_model, d_pos= max(d_model, depth*width), depth= depth, width= width, d_add_pos=32)

        # transformer encoder
        self.encoder = EnhancedTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            dropout=dropout,
            prop_dim=prop_dim,
            use_gnn=use_gnn
        )
          # transformer encoder
        # encoder_layer = nn.TransformerEncoderLayer(d_model= d_model, nhead= nhead, dim_feedforward= d_ff,
        #                                            dropout= self.dropout, activation= activation, batch_first= True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)

        # vae
        self.fc_vae = nn.Sequential(nn.Linear(d_model, latent_dim), 
                                    nn.Linear(latent_dim, 2*latent_dim))

        # transformer decoder
        self.fc_memory = nn.Linear(latent_dim, d_model)

        self.decoder = EnhancedTransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            dropout=dropout,
            prop_dim=prop_dim,
            use_gnn=False
        )
       
        self.fc_dec = nn.Linear(d_model, num_tokens)

        # for decode smiles
        self.labels = None
    
    def sequential_decode(self, z: torch.Tensor, frag_ecfps: torch.Tensor, ndummys: torch.Tensor, 
                      max_nfrags: int = 30, free_n: bool = False, asSmiles: bool = False, 
                      conditions: torch.Tensor = None,  sampling: bool = False) -> tuple:  
        """
        ...（原有 docstring 不变）...
        Returns:
            如果 asSmiles=True: (list of SMILES, log_probs Tensor)
            否则: (list of FragmentTree, log_probs Tensor)
        """
        batch_size = z.shape[0]
        device = z.device
        num_conditions = conditions.shape[1] if conditions is not None else 0

       
        log_probs = torch.zeros(batch_size, device=device)

        # latent variable to memory
        memory = self.fc_memory(z).unsqueeze(1)

        # root prediction
        root_embed = self.embed(torch.zeros(batch_size, 1, device=device).long())
        if num_conditions > 0:
            root_embed = torch.cat([conditions, root_embed], dim=1)
        tgt_pad_mask = torch.all(root_embed == 0, dim=-1).to(device)
        out = self.decoder(root_embed, memory, tgt_key_padding_mask=tgt_pad_mask)
        out = self.fc_dec(out[:, num_conditions:])
        
        # root  log prob
        root_logits = out.squeeze(1)  # (B, num_labels)

        root_log_probs = torch.log_softmax(root_logits, dim=-1)  # (B, num_labels)

        if sampling:
            root_idxs = torch.multinomial(torch.exp(root_log_probs), 1).squeeze(-1)
        else:
            root_idxs = root_log_probs.argmax(dim=-1)

        log_probs += root_log_probs[torch.arange(batch_size), root_idxs]

        # root_probs = torch.softmax(root_logits, dim=-1)   # torch.log_softmax
        # if sampling:
        #     sampled_probs = root_probs + 1e-8
        #     # sampled_probs = sampled_probs / sampled_probs.sum(dim=-1, keepdim=True)  
        #     idxs = torch.multinomial(sampled_probs, 1).squeeze(-1)
        #     # sampled_probs（或原始 probs，如果没加 1e-8）
        #     log_probs = torch.log(sampled_probs[torch.arange(B), idxs] + 1e-12)
        # else:
        #     root_idxs = root_probs.argmax(dim=-1).flatten()  # (B,)
        #     log_probs += torch.log(root_probs[torch.arange(batch_size), root_idxs] + 1e-8)

        # root_idxs = out.argmax(dim=-1).flatten()  # (B,)
        # log_probs += torch.log(root_probs[torch.arange(batch_size), root_idxs] + 1e-8)

        continues = []
        target_ids = [0] * batch_size
        target_ids_list = [[0] for _ in range(batch_size)]
        tree_list = [FragmentTree() for _ in range(batch_size)]
        for i, idx in enumerate(root_idxs):
            parent_id = tree_list[i].add_node(parent_id=None, feature=frag_ecfps[idx], fid=idx.item(), bondtype=0)
            assert parent_id == 0
            tree_list[i].set_positional_encoding(parent_id, d_pos=self.depth * self.width)
            continues.append(ndummys[idx].item() > 0)

        nfrags = 1
        while (nfrags < max_nfrags) & (sum(continues) > 0):
            tgt_mask = generate_square_subsequent_mask(length=nfrags + num_conditions + 1).to(device)
            tgt_mask[:, :num_conditions + 1] = 0
            tgt_pad_mask = torch.hstack([tgt_pad_mask, tgt_pad_mask.new_full(size=(batch_size, 1), fill_value=False)])
            features = get_pad_features(tree_list, key='x', max_nodes_num=nfrags).to(device)
            positions = get_pad_features(tree_list, key='pos', max_nodes_num=nfrags).to(device)
            assert features.shape[0] == positions.shape[0]

            tgt = self.fc_ecfp(features) + self.PE(positions)
            tgt = torch.cat([root_embed, tgt], dim=1)

            out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
            out = self.fc_dec(out[:, num_conditions:])  # (B, nfrags+1, num_labels)

            # === 修改：计算当前步的 log prob ===
            step_logits = out[:, -1, :]  # (B, num_labels)
            
            step_log_probs = torch.log_softmax(step_logits, dim=-1)  # (B, num_labels)

            if sampling:
                new_idxs = torch.multinomial(torch.exp(step_log_probs), 1).squeeze(-1)
            else:
                new_idxs = step_log_probs.argmax(dim=-1)

            # 修正：强制无效 fragment 转为终止符 (idx=0)，并修正 logp
            # for i in range(batch_size):
            #     if continues[i]:
            #         idx = new_idxs[i].item()
            #         if idx != 0 and ndummys[idx].item() == 0:
            #             new_idxs[i] = 0
            #             # logp 改为选择终止符的概率
            #             step_log_probs[i, :] = torch.log_softmax(step_logits[i], dim=-1)  # 确保最新
            #             # 实际上这里 step_log_probs[i, 0] 就是终止符 logp

            # 累积 logp（仅 active）
            active_mask = torch.tensor(continues, device=device, dtype=torch.bool)
            step_logp = step_log_probs[torch.arange(batch_size), new_idxs]
            log_probs += step_logp * active_mask.float()


            # step_probs = torch.softmax(step_logits, dim=-1)

            # if sampling:
            #     step_probs = step_probs + 1e-8
            #     # step_probs = probs / probs.sum(dim=-1, keepdim=True)
            #     new_idxs = torch.multinomial(step_probs, 1).squeeze(-1)
            # else:
            #     new_idxs = step_probs.argmax(dim=-1).flatten()

            # # new_idxs = step_logits.argmax(dim=-1).flatten()  # (B,)

            # # 仅对仍在生成的样本累积 logp
            # active_mask = torch.tensor(continues, device=device, dtype=torch.bool)
            # step_logp = torch.log(step_probs[torch.arange(batch_size), new_idxs] + 1e-8)
            # log_probs += step_logp * active_mask.float()

            # add node 
            for i, idx in enumerate(new_idxs):
                if continues[i]:
                    if ndummys[idx] == 0:
                        idx = torch.tensor(0)
                    if idx != 0:
                        parent_id = target_ids[i]
                        add_node_id = tree_list[i].add_node(parent_id=parent_id, feature=frag_ecfps[idx], fid=idx.item(), bondtype=1)
                        parent_fid = tree_list[i].dgl_graph.ndata['fid'][parent_id].item()
                        num_sibling = ndummys[parent_fid].item() - 1 if parent_id > 0 else ndummys[parent_fid].item()
                        if free_n:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling=num_sibling, d_pos=self.depth * self.width)
                        else:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling=self.width, d_pos=self.depth * self.width)
                        level = tree_list[i].dgl_graph.ndata['level'][add_node_id].item()

                        if len(tree_list[i].dgl_graph.predecessors(parent_id)) >= num_sibling:
                            target_ids_list[i].pop(-1)

                        if (ndummys[idx] > 1) & (self.depth > level):
                            target_ids_list[i].append(add_node_id)

                    continues[i] = bool(target_ids_list[i]) if (idx != 0) else False
                    target_ids[i] = target_ids_list[i][-1] if continues[i] else 0
            nfrags += 1

        if asSmiles:
            if self.labels is not None:
                outputs = Parallel(n_jobs=self.n_jobs)(
                    delayed(constructMol)(
                        self.labels[tree.dgl_graph.ndata['fid'].squeeze(-1).tolist()],
                        tree.adjacency_matrix().tolist()
                    ) for tree in tree_list
                )
            else:
                raise ValueError('If asSmiles= True, please set labels. example; self.set_labels(labels)')
        else:
            outputs = tree_list

        return outputs, log_probs 
  

class TreeGNNLayer(nn.Module):
    def __init__(self, d_model, edge_dim=3, heads=1):
        super().__init__()
        self.gnn = GATv2Conv(
            in_channels=d_model,
            out_channels=d_model,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)
        self.res_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, D]
        edge_index: [2, E]
        edge_attr: [E, 3]
        """
        residual = x
        out = self.gnn(x, edge_index, edge_attr)
        out = residual + self.res_weight * out
        out = self.norm(out)
        return out