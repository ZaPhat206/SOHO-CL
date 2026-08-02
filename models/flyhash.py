import torch
import torch.nn as nn

class FlyHash(nn.Module):
    def __init__(self, in_dim: int, expand_dim: int, synaptic_degree: int):
        super().__init__()
        self.in_dim = in_dim
        self.expand_dim = expand_dim
        self.synaptic_degree = synaptic_degree
        
        # Initialize sparse projection matrix
        proj = torch.zeros(expand_dim, in_dim)
        for row in range(expand_dim):
            selected_cols = torch.randperm(in_dim)[:synaptic_degree]
            proj[row, selected_cols] = torch.randn(synaptic_degree)
        self.register_buffer("projection_matrix", proj)
        
    def to_sparse(self):
        self.projection_matrix = self.projection_matrix.to_sparse_csc()

    def forward(self, x, coding_level: float, absolute_wta: bool = False):
        """
        x: (N, in_dim)
        Returns sparse activated features (N, expand_dim)
        """
        # x is transposed for matmul if sparse
        if self.projection_matrix.is_sparse:
            features = torch.sparse.mm(self.projection_matrix, x.T) # (expand_dim, N)
        else:
            features = self.projection_matrix @ x.T
            
        k = int(self.expand_dim * coding_level)
        if absolute_wta:
            values, indices = torch.abs(features).topk(k, dim=0, largest=True)
            # Retrieve original signs
            original_values = features.gather(0, indices)
            output = torch.zeros_like(features)
            output.scatter_(0, indices, original_values)
        else:
            values, indices = features.topk(k, dim=0, largest=True)
            output = torch.zeros_like(features)
            output.scatter_(0, indices, values)
            
        return output.T # (N, expand_dim)
