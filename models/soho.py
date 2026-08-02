import torch
import torch.nn as nn

class IncrementalOLDA:
    def __init__(self, in_dim: int, device: torch.device):
        self.in_dim = in_dim
        self.device = device
        
        self.class_sums = {}
        self.class_counts = {}
        self.global_sum = torch.zeros(in_dim, device=device)
        self.global_count = 0
        
        self.S_w = torch.zeros(in_dim, in_dim, device=device)
        
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        """Update scatter matrices with new data."""
        unique_classes = torch.unique(labels)
        for c in unique_classes:
            c = c.item()
            mask = (labels == c)
            class_features = features[mask]
            
            n_c = class_features.shape[0]
            sum_c = class_features.sum(dim=0)
            mu_c = sum_c / n_c
            
            if c not in self.class_sums:
                self.class_sums[c] = sum_c
                self.class_counts[c] = n_c
            else:
                self.class_sums[c] += sum_c
                self.class_counts[c] += n_c
                
            # Update Within-class scatter incrementally (Assuming CIL disjoint classes)
            centered = class_features - mu_c
            self.S_w += centered.T @ centered
            
        self.global_sum += features.sum(dim=0)
        self.global_count += features.shape[0]
        
    def compute_projection(self, output_dim: int):
        """Compute the Orthogonal LDA projection matrix R."""
        mu_global = self.global_sum / self.global_count
        S_b = torch.zeros_like(self.S_w)
        
        for c, sum_c in self.class_sums.items():
            n_c = self.class_counts[c]
            mu_c = sum_c / n_c
            diff = (mu_c - mu_global).unsqueeze(1)
            S_b += n_c * (diff @ diff.T)
            
        # Add regularization to avoid singular matrix
        S_w_reg = self.S_w + 1e-4 * torch.eye(self.in_dim, device=self.device)
        
        # Solve generalized eigenvalue problem: S_b v = lambda S_w v => S_w^{-1} S_b v = lambda v
        inv_S_w = torch.linalg.inv(S_w_reg)
        target_matrix = inv_S_w @ S_b
        
        eigenvalues, eigenvectors = torch.linalg.eig(target_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        # Sort eigenvalues and take top ones
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        # Cap output_dim to in_dim since we can't extract more eigenvectors than the dimension
        actual_out_dim = min(output_dim, self.in_dim)
        top_indices = sorted_indices[:actual_out_dim]
        V = eigenvectors[:, top_indices] # (in_dim, actual_out_dim)
        
        # Orthogonalize via SVD
        U, S, Vh = torch.linalg.svd(V, full_matrices=False)
        R = U # (in_dim, actual_out_dim)
        
        return R.T # (actual_out_dim, in_dim)


class SOHO(nn.Module):
    def __init__(self, in_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.device = device
        self.olda = IncrementalOLDA(in_dim, device)
        
        # Store current projection matrix R
        self.R = torch.randn(min(output_dim, in_dim), in_dim, device=device)
        
    def update_stats(self, features: torch.Tensor, labels: torch.Tensor):
        self.olda.update(features, labels)
        self.R = self.olda.compute_projection(self.output_dim)
        
    def forward(self, x: torch.Tensor, coding_level: float, absolute_wta: bool = False):
        """
        x: (N, in_dim)
        Returns sparse activated features (N, output_dim)
        """
        z = x @ self.R.T # (N, actual_out_dim)
        
        k = int(z.shape[1] * coding_level)
        if absolute_wta:
            values, indices = torch.abs(z).topk(k, dim=1, largest=True)
            original_values = z.gather(1, indices)
            output = torch.zeros_like(z)
            output.scatter_(1, indices, original_values)
        else:
            values, indices = z.topk(k, dim=1, largest=True)
            output = torch.zeros_like(z)
            output.scatter_(1, indices, values)
            
        return output
