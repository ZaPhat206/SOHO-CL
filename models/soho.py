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
        # Spherical OLDA (L2-Normalized OLDA)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        unique_classes = torch.unique(labels)

        # FIX Hướng 3: Welford's Parallel Covariance Formula cho S_w
        # Công thức chuẩn xác để cộng dồn scatter khi mean thay đổi theo từng task:
        #   S_w_combined = S_w_old + S_batch + (n_old*n_new)/(n_old+n_new) * outer(mu_old - mu_new)
        # Đây là cách duy nhất đúng về mặt toán học khi không lưu lại dữ liệu cũ.
        for c in unique_classes:
            c = c.item()
            mask = (labels == c)
            class_features = features[mask]
            n_new = class_features.shape[0]
            sum_new = class_features.sum(dim=0)
            mu_new = sum_new / n_new  # Mean của batch hiện tại

            # Scatter của batch hiện tại quanh mean của chính batch đó
            centered_new = class_features - mu_new
            S_batch = centered_new.T @ centered_new

            if c not in self.class_sums:
                # Lần đầu xuất hiện: chưa có dữ liệu cũ, không có correction term
                self.S_w += S_batch
                self.class_sums[c] = sum_new
                self.class_counts[c] = n_new
            else:
                # Đã có dữ liệu cũ: thêm correction term (Welford's)
                n_old = self.class_counts[c]
                mu_old = self.class_sums[c] / n_old  # Mean cũ TRƯỚC khi cập nhật
                
                # Số hạng hiệu chỉnh: bù đắp cho sự dịch chuyển của mean
                delta = (mu_new - mu_old).unsqueeze(1)  # (D, 1)
                correction = (n_old * n_new) / (n_old + n_new) * (delta @ delta.T)
                
                self.S_w += S_batch + correction
                self.class_sums[c] += sum_new
                self.class_counts[c] += n_new

        self.global_sum += features.sum(dim=0)
        self.global_count += features.shape[0]
        
    def compute_projection(self, output_dim: int):
        n_total = sum(self.class_counts.values())

        mu_global = self.global_sum / self.global_count
        S_b = torch.zeros_like(self.S_w)

        for c, sum_c in self.class_sums.items():
            n_c = self.class_counts[c]
            mu_c = sum_c / n_c
            diff = (mu_c - mu_global).unsqueeze(1)
            S_b += n_c * (diff @ diff.T)

        # FIX Bug#2: Normalize scatter matrices theo tổng số mẫu
        # S_w tích lũy qua nhiều tasks → scale tăng liên tục → inv(S_w) → gần 0
        # → eigenvectors của inv_S_w @ S_b mất ý nghĩa phân loại.
        # Normalize bằng n_total → giá trị ổn định bất kể đã học bao nhiêu tasks.
        S_w_norm = self.S_w / n_total
        S_b_norm = S_b / n_total

        S_w_reg = S_w_norm + 1e-4 * torch.eye(self.in_dim, device=self.device)

        # FIX Bug#4: GEVD qua Cholesky Transform — numerically stable
        # Bài toán: S_b v = λ S_w v (Generalized Eigenvalue Problem)
        # Cách cũ: inv(S_w) @ S_b → eig() → unstable vì non-symmetric + dùng inv()
        # Cách mới: Cholesky L s.t. S_w = L @ L^T → C = L^{-1} @ S_b @ L^{-T}
        #           C là symmetric → eigh(C) stable → v = L^{-T} @ w
        try:
            L_chol  = torch.linalg.cholesky(S_w_reg)       # S_w_reg = L @ L^T
            L_inv   = torch.linalg.inv(L_chol)             # Triangular inverse O(D²)
            C       = L_inv @ S_b_norm @ L_inv.T
            C       = (C + C.T) / 2                        # Force exact symmetry

            eigenvalues, eigvecs_C = torch.linalg.eigh(C)  # Ascending order
            eigenvalues  = eigenvalues.flip(0)              # → Descending
            eigvecs_C    = eigvecs_C.flip(1)

            # Chuyển về không gian gốc: v = L^{-T} @ w, normalize cột
            eigenvectors = L_inv.T @ eigvecs_C
            col_norms    = torch.linalg.norm(eigenvectors, dim=0, keepdim=True).clamp(min=1e-8)
            eigenvectors = eigenvectors / col_norms

        except Exception:
            # Fallback: phương pháp cũ nếu Cholesky thất bại (S_w_reg không PD)
            S_w_fb  = S_w_norm + 1e-3 * torch.eye(self.in_dim, device=self.device)
            inv_S_w = torch.linalg.inv(S_w_fb)
            target  = inv_S_w @ S_b_norm
            ev, evec = torch.linalg.eig(target)
            eigenvalues  = ev.real
            eigenvectors = evec.real
            idx          = torch.argsort(eigenvalues, descending=True)
            eigenvalues  = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # =========================================================================
        # NSP-OLDA: Discriminative Subspace
        # =========================================================================
        # S_b có hạng tối đa N-1 (N = số class đã thấy). Lấy top N-1 eigenvectors.
        N = len(self.class_sums)
        num_disc = min(N - 1, self.in_dim)

        if num_disc > 0:
            V_disc = eigenvectors[:, :num_disc]
            Q_disc, _ = torch.linalg.qr(V_disc)    # Trực giao hóa Discriminative Subspace
        else:
            Q_disc = torch.empty((self.in_dim, 0), device=self.device)

        # =====================================================================
        # ETF PROCRUSTES ALIGNMENT
        # =====================================================================
        if self.use_etf and N > 1 and Q_disc.shape[1] == N - 1:
            # BƯỚC 1: Tâm điểm lớp trong không gian OLDA
            mu_global_etf = self.global_sum / self.global_count
            M_orig_list = []
            for c in sorted(self.class_sums.keys()):
                mu_c = (self.class_sums[c] / self.class_counts[c]) - mu_global_etf
                M_orig_list.append(mu_c.unsqueeze(1))

            M_orig = torch.cat(M_orig_list, dim=1)          # (D, N)
            M      = Q_disc.T @ M_orig                      # (N-1, N)
            M_norm = torch.nn.functional.normalize(M, p=2, dim=0)

            # BƯỚC 2: ETF lý tưởng E
            I_N    = torch.eye(N, device=self.device)
            Ones_N = torch.ones(N, N, device=self.device)
            P      = I_N - (1.0 / N) * Ones_N
            U_P, _, _ = torch.linalg.svd(P)
            U_sub  = U_P[:, :N-1]                           # (N, N-1)
            E      = ((N / (N - 1)) ** 0.5) * U_sub.T      # (N-1, N)

            # BƯỚC 3: Orthogonal Procrustes → xoay Q_disc về gần ETF nhất
            U_proc, _, Vh_proc = torch.linalg.svd(M_norm @ E.T)
            Q_rot  = U_proc @ Vh_proc                       # (N-1, N-1)
            Q_disc = Q_disc @ Q_rot
        # =====================================================================

        # Null-Space Preservation
        num_null_dims = self.in_dim - Q_disc.shape[1]

        if num_null_dims > 0:
            I_d     = torch.eye(self.in_dim, device=self.device)
            P_ortho = I_d - Q_disc @ Q_disc.T
            # eigh trên symmetric PSD matrix: nhanh hơn SVD 3-5x
            eigvals, eigvecs = torch.linalg.eigh(P_ortho)
            Q_null  = eigvecs[:, -num_null_dims:]           # Eigenvalues ≈ 1 = null space
            R_full  = torch.cat([Q_disc, Q_null], dim=1)
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
        
        # Ma trận Sparse Rademacher (-1, 0, 1)
        # REVERT JL scaling: scaling 1/sqrt(d*D) làm G nhỏ đi ~76x, khiến
        # lambda tối ưu dịch ra ngoài search range của GCV → accuracy sụt.
        # GCV đã tự thích nghi với scale của G qua SVD → không cần JL scaling.
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
        # 0. Spherical OLDA: Chuẩn hóa L2 đầu vào (chỉ input, không phải sau WTA)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        
        # 1. Chiếu trực giao: z = x @ R^T
        z = x_norm @ self.R.T # (N, olda_dim)
        
        # 2. Mở rộng chiều: v = W @ z^T -> shape (output_dim, N)
        # Dùng chiều (output_dim, N) giống hệt FLY baseline (expand_dim, N)
        expanded = self.W @ z.T  # (output_dim, N)
        
        # 3. Áp dụng WTA theo dim=0 (đúng chuẩn FLY-CL: Winner-Takes-All theo chiều Feature)
        k = max(1, int(expanded.shape[0] * coding_level))
        if absolute_wta:
            values, indices = torch.abs(expanded).topk(k, dim=0, largest=True)
            original_values = expanded.gather(0, indices)
            output = torch.zeros_like(expanded)
            output.scatter_(0, indices, original_values)
        else:
            values, indices = expanded.topk(k, dim=0, largest=True)
            output = torch.zeros_like(expanded)
            output.scatter_(0, indices, values)
            
        return output.T  # Trả về shape (N, output_dim) cho Ridge Regression
