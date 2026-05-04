import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ListDataset(Dataset):
    def __init__(self, frag_indices: list , positions: list, prop: torch.Tensor) -> None:
        """
        frag_indices, positions: list of torch.Tensors with different lengths
        ecfps, prop: torch.Tensor
        """
        super().__init__()
        self.ecfps = None
        self.frag_indices = frag_indices
        self.positions = positions
        self.prop = prop
        

    def __len__(self):
        return len(self.frag_indices)
    
    def __getitem__(self, index) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.ecfps is None:
            return self.frag_indices[index], self.positions[index], self.prop[index]
        else:
            return self.ecfps[index], self.frag_indices[index], self.positions[index], self.prop[index]

    def set_stereo(self, ecfps):
        self.ecfps = ecfps


def collate_pad_fn(batch):
    frag_indices, positions, props = zip(*batch)
    frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
    positions = pad_sequence(positions, batch_first= True, padding_value= 0)
    props = torch.stack(props)

    return frag_indices, positions, props

def collate_stereo_fn(batch):
    ecfps, frag_indices, positions, props = zip(*batch)
    ecfps = torch.stack(ecfps)
    frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
    positions = pad_sequence(positions, batch_first= True, padding_value= 0)
    props = torch.stack(props)

    return ecfps, frag_indices, positions, props


# class ListDatasetEdgeIndex(ListDataset):
#     def __init__(self, frag_indices: list , positions: list, prop: torch.Tensor, edge_index: torch.Tensor) -> None:
#         super().__init__(frag_indices, positions, prop)
#         self.edge_index = edge_index

#     def __getitem__(self, index) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         if self.ecfps is None and self.edge_index is None:
#             return self.frag_indices[index], self.positions[index], self.prop[index]
#         elif self.edge_index is not None:
#              return self.frag_indices[index], self.positions[index], self.prop[index], self.edge_index[index]
#         else:
#             return self.ecfps[index], self.frag_indices[index], self.positions[index], self.prop[index]
        

# def collate_pad_fn(batch):
#         frag_indices, positions, props, edge_index = zip(*batch)
#         frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
#         positions = pad_sequence(positions, batch_first= True, padding_value= 0)
#         props = torch.stack(props)
#         edge_index = torch.stack(edge_index)

#         return frag_indices, positions, props, edge_index

# def collate_stereo_fn(batch):
#         ecfps, frag_indices, positions, props, edge_index= zip(*batch)
#         ecfps = torch.stack(ecfps)
#         frag_indices = pad_sequence(frag_indices, batch_first= True, padding_value= 0)
#         positions = pad_sequence(positions, batch_first= True, padding_value= 0)
#         props = torch.stack(props)
#         edge_index = torch.stack(edge_index)

#         return ecfps, frag_indices, positions, props, edge_index

