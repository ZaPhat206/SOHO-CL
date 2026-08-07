import torch
import torch.nn as nn

class IncrementalOLDA:
    def __init__(self, in_dim: int, device: torch.device, use_etf: bool = True):
        self.in_dim = in_dim
        self.device = device
        self.use_etf = use_etf
        
        self.class_sums = {}
        self.class_counts = {}
        self.global_sum = torch.zeros(in_dim, device=device)
        self.global_count = 0
        
        self.S_w = torch.zeros(in_dim, in_dim, device=device)
        
    def update(self, features: torch.Tensor, labels: torch.Tensor):
        # ĐỘT PHÁ TOÁN HỌC: Spherical OLDA (L2-Normalized OLDA)
        # Bóp các mẫu cùng lớp thành khối cầu đặc kịt trước khi tính phương sai
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
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
        
        # =====================================================================
        # ĐỘT PHÁ TOÁN HỌC - ETF PROCRUSTES ALIGNMENT
        # =====================================================================
        N = len(self.class_sums)
        # Chỉ kích hoạt khi số class > 1 và số chiều Q_disc khớp chính xác N-1
        if self.use_etf and N > 1 and Q_disc.shape[1] == N - 1:
            # BƯỚC 1: Trích xuất Tâm điểm M trong không gian OLDA
            mu_global = self.global_sum / self.global_count
            M_orig_list = []
            for c in sorted(self.class_sums.keys()):
                mu_c = (self.class_sums[c] / self.class_counts[c]) - mu_global
                M_orig_list.append(mu_c.unsqueeze(1))
            
            # M_orig: (768, N) -> M: (N-1, N)
            M_orig = torch.cat(M_orig_list, dim=1)
            M = Q_disc.T @ M_orig
            
            # Chuẩn hóa M để xoay góc thuần túy (không bị lệch do độ dài)
            M_norm = torch.nn.functional.normalize(M, p=2, dim=0)
            
            # BƯỚC 2: Sinh bộ khung ETF lý tưởng E
            I_N = torch.eye(N, device=self.device)
            Ones_N = torch.ones(N, N, device=self.device)
            P = I_N - (1.0 / N) * Ones_N
            
            U_P, _, _ = torch.linalg.svd(P)
            U_sub = U_P[:, :N-1] # (N, N-1)
            E = ((N / (N - 1)) ** 0.5) * U_sub.T # E: (N-1, N)
            
            # BƯỚC 3: Thuật toán Orthogonal Procrustes
            # Tìm ma trận xoay Q trực giao để M_norm khớp vào E
            U_proc, _, Vh_proc = torch.linalg.svd(M_norm @ E.T)
            Q_rot = U_proc @ Vh_proc # Ma trận xoay: (N-1, N-1)
            
            # Xoay toàn bộ trục không gian Discriminative (Áp dụng xoay vào Q_disc)
            Q_disc = Q_disc @ Q_rot
        # =====================================================================
        
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
    def __init__(self, in_dim: int, output_dim: int, device: torch.device, density: float = 0.3, olda_dim: int = 768, use_etf: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.output_dim = output_dim
        self.device = device
        
        self.olda_dim = min(olda_dim, in_dim)
        self.olda = IncrementalOLDA(in_dim, device, use_etf=use_etf)
        
        # Khởi tạo R bằng Ma trận Đơn vị (Identity Matrix) cho Task 1.
        self.R = torch.eye(self.olda_dim, in_dim, device=device)
        
        # ĐỘT PHÁ TOÁN HỌC: Ma trận Sparse Rademacher (-1, 0, 1)
        # Khắc phục triệt để "Gaussian Annihilation" (sự hủy diệt của cấu trúc trực giao)
        # Giữ nguyên tỷ lệ thưa để bảo toàn khoảng cách JL.
        random_tensor = torch.rand(self.output_dim, self.olda_dim, device=device)
        self.W = torch.zeros(self.output_dim, self.olda_dim, device=device)
        self.W[random_tensor < (density / 2)] = 1.0
        self.W[(random_tensor >= (density / 2)) & (random_tensor < density)] = -1.0
        
    def update_stats(self, features: torch.Tensor, labels: torch.Tensor):
        self.olda.update(features, labels)
        self.R = self.olda.compute_projection(self.olda_dim)
        
    def forward(self, x: torch.Tensor, coding_level: float, absolute_wta: bool = False):
        """
        x: (N, in_dim)
        Returns sparse activated features (N, output_dim)
        """
        # 0. Spherical OLDA: Chuẩn hóa L2 đầu vào
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        
        # 1. Chiếu trực giao: z = x @ R^T
        z = x_norm @ self.R.T # (N, olda_dim)
        
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
