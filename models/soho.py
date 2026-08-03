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
                
            centered = class_features - mu_c
            self.S_w += centered.T @ centered
            
        self.global_sum += features.sum(dim=0)
        self.global_count += features.shape[0]
        
    def compute_projection(self, output_dim: int):
        mu_global = self.global_sum / self.global_count
        S_b = torch.zeros_like(self.S_w)
        
        for c, sum_c in self.class_sums.items():
            n_c = self.class_counts[c]
            mu_c = sum_c / n_c
            diff = (mu_c - mu_global).unsqueeze(1)
            S_b += n_c * (diff @ diff.T)
            
        S_w_reg = self.S_w + 1e-4 * torch.eye(self.in_dim, device=self.device)
        
        inv_S_w = torch.linalg.inv(S_w_reg)
        target_matrix = inv_S_w @ S_b
        
        eigenvalues, eigenvectors = torch.linalg.eig(target_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        # =========================================================================
        # ĐỘT PHÁ TOÁN HỌC: NSP-OLDA (Null-Space Preserving Orthogonal LDA)
        # =========================================================================
        # Sắp xếp trị riêng giảm dần để lấy các trục phân loại mạnh nhất lên đầu
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 1. Trích xuất không gian đặc trưng có ý nghĩa (Discriminative Subspace)
        positive_mask = eigenvalues > 1e-5
        if not positive_mask.any():
            return torch.eye(self.in_dim, device=self.device)
            
        V_disc = eigenvectors[:, positive_mask]
        
        # Trực giao hóa không gian Discriminative
        Q_disc, _ = torch.linalg.qr(V_disc) 
        
        # 2. Bảo toàn Không gian Rỗng (Null Space)
        num_null_dims = self.in_dim - Q_disc.shape[1]
        
        if num_null_dims > 0:
            I = torch.eye(self.in_dim, device=self.device)
            P_ortho = I - Q_disc @ Q_disc.T
            
            U_null, S_null, _ = torch.linalg.svd(P_ortho)
            Q_null = U_null[:, :num_null_dims]
            
            R_full = torch.cat([Q_disc, Q_null], dim=1)
        else:
            R_full = Q_disc
            
        actual_out_dim = min(output_dim, self.in_dim)
        R = R_full[:, :actual_out_dim].T
        return R


class SOHO(nn.Module):
    def __init__(self, in_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.device = device
        
        # NSP-OLDA: Giữ nguyên 100% số chiều (768D), KHÔNG vứt bỏ thông tin.
        self.olda_dim = in_dim
        self.olda = IncrementalOLDA(in_dim, device)
        
        # Khởi tạo R bằng Ma trận Đơn vị (Identity Matrix) cho Task 1.
        self.R = torch.eye(self.olda_dim, in_dim, device=device)
        
        # Cốt lõi của SOHO: Ma trận mở rộng NHỊ PHÂN THƯA (10%)
        # ĐÃ PHÁT HIỆN LỖI: Ma trận Nhị phân (0, 1) không có số âm, làm phá hủy sự phân phối khoảng cách
        # theo định lý Johnson-Lindenstrauss. 
        # NÂNG CẤP LÊN DENSE GAUSSIAN (Sức mạnh Toán học tối thượng)
        self.W = torch.randn(self.output_dim, self.olda_dim, device=device)
        
    def update_stats(self, features: torch.Tensor, labels: torch.Tensor):
        self.olda.update(features, labels)
        self.R = self.olda.compute_projection(self.olda_dim)
        
    def forward(self, x: torch.Tensor, coding_level: float, absolute_wta: bool = False):
        """
        x: (N, in_dim)
        Returns sparse activated features (N, output_dim)
        """
        # 1. Chiếu trực giao: z = x @ R^T
        z = x @ self.R.T # (N, olda_dim)
        
        # 2. Mở rộng chiều: v = z @ W^T
        expanded = z @ self.W.T # (N, output_dim)
        
        # 3. Áp dụng WTA trên không gian mở rộng
        k = max(1, int(expanded.shape[1] * coding_level))
        if absolute_wta:
            values, indices = torch.abs(expanded).topk(k, dim=1, largest=True)
            original_values = expanded.gather(1, indices)
            output = torch.zeros_like(expanded)
            output.scatter_(1, indices, original_values)
        else:
            values, indices = expanded.topk(k, dim=1, largest=True)
            output = torch.zeros_like(expanded)
            output.scatter_(1, indices, values)
            
        return output
