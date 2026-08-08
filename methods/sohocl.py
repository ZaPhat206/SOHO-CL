import torch
import numpy as np
from methods.base_cl import BaseCL
from models.soho import SOHO
from utils.train_utils import feature_extract, target2onehot

def select_ridge_parameter(Features, Y, ridge_lower, ridge_upper):
    X = Features
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_sq = S**2
    UTY = U.T @ Y
    ridges = torch.tensor(10.0 ** np.arange(ridge_lower, ridge_upper))
    n_samples = X.shape[0]
    
    gcv_scores = []
    for ridge in ridges:
        diag = S_sq / (S_sq + ridge)
        df = diag.sum()
        Y_hat = U @ (diag[:, None] * UTY)
        residual = torch.norm(Y - Y_hat)**2
        
        # FIX Lỗi 3: Không clamp denom, dùng công thức chuẩn như FLY-CL
        gcv = (residual / n_samples) / (1 - df / n_samples)**2
        gcv_scores.append(gcv.item())

    optimal_idx = np.argmin(gcv_scores)
    return ridges[optimal_idx]

class SOHOCL(BaseCL):
    def __init__(self, backbone, soho: SOHO, num_classes: int, coding_level: float, 
                 ridge_lower: float, ridge_upper: float, device: torch.device):
        super().__init__(backbone, device)
        self.soho = soho.to(device)
        self.num_classes = num_classes
        self.coding_level = coding_level
        self.ridge_lower = ridge_lower
        self.ridge_upper = ridge_upper
        
        self.Wo = None
        
        # In SOHO, since projection matrix R changes dynamically per task, 
        # and WTA is a non-linear operation, we must store the backbone features 
        # to re-project and re-solve Ridge accurately at each step.
        self.memory_features = []
        self.memory_labels = []

    def train_task(self, task_id: int, train_loader):
        import time
        training_start = time.time()
        
        feature_extract_start = time.time()
        # 1. Extract features for new task
        new_embeddings, new_labels = feature_extract(self.backbone, train_loader, self.device)
        feature_extract_end = time.time()
        extract_time = feature_extract_end - feature_extract_start
        
        # 2. Update SOHO OLDA statistics incrementally with NEW data
        self.soho.update_stats(new_embeddings, new_labels)
        
        # 3. Add to memory buffer (768D features are very lightweight)
        self.memory_features.append(new_embeddings)
        self.memory_labels.append(new_labels)
        
        all_features = torch.cat(self.memory_features, dim=0)
        all_labels = torch.cat(self.memory_labels, dim=0)
        
        # FIX Lỗi 2: Bỏ Label Smoothing. Ridge Regression tự chống Overfit qua lambda rồi.
        # Thêm Label Smoothing vào chỉ khiến mô hình bị phạt 2 lần -> Under-learning trên CUB/CIFAR.
        Y = target2onehot(all_labels, self.num_classes)
        
        # =========================================================================
        # ĐỘT PHÁ TỐI ƯU MEMORY: Xử lý Chunking (Mini-batching)
        # Thay vì tống 50,000 ảnh (2GB) vào RAM cùng lúc gây tràn bộ nhớ, 
        # ta chia nhỏ ra xử lý từng cụm 2000 ảnh. RAM sẽ tụt xuống chỉ còn ~80MB!
        # =========================================================================
        Q_global = torch.zeros(self.soho.output_dim, self.num_classes, device=self.device)
        G_global = torch.zeros(self.soho.output_dim, self.soho.output_dim, device=self.device)
        
        chunk_size = 2000
        n_samples = all_features.shape[0]
        n_new = new_embeddings.shape[0]
        start_new = n_samples - n_new
        
        z_sparse_new_list = []
        
        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                feat_chunk = all_features[i:end]
                Y_chunk = Y[i:end]
                
                # FIX Lỗi 7: Bỏ L2 normalize sau WTA - FLY-CL không có bước này.
                # Normalize phá vỡ tnh thưa (Sparsity) và mất thông tin cường độ kích hoạt.
                z_chunk = self.soho(feat_chunk, self.coding_level, absolute_wta=False)
                
                # Cộng dồn ma trận
                Q_global += z_chunk.T @ Y_chunk
                G_global += z_chunk.T @ z_chunk
                
                # Trích xuất riêng dữ liệu của Task mới nhất để tính Lambda
                if end > start_new:
                    chunk_start_in_new = max(0, start_new - i)
                    z_sparse_new_list.append(z_chunk[chunk_start_in_new:])
                
        z_sparse_new = torch.cat(z_sparse_new_list, dim=0)
        Y_new = Y[start_new:]
        
        # FIX Lỗi GCV-G_global Mismatch:
        # GCV chọn lambda trên z_sparse_new (chỉ Task T, n_new mẫu).
        # Nhưng Cholesky dùng G_global (tất cả tasks, n_all mẫu).
        # Lambda tối ưu cho G_global lớn hơn ~(n_all/n_new) lần.
        # Scale lambda để bù đắp sự chệnh lệch này.
        best_lam = select_ridge_parameter(z_sparse_new, Y_new, self.ridge_lower, self.ridge_upper)
        scale_factor = n_samples / n_new  # n_all / n_new
        scaled_lam = best_lam * scale_factor
        
        # Chỉ dùng scaled_lam, bỏ 1e-6 dư thừa (FLY-CL không có bước này)
        G_reg = G_global + scaled_lam * torch.eye(G_global.size(0), device=self.device)
        L = torch.linalg.cholesky(G_reg)
        self.Wo = torch.cholesky_solve(Q_global, L)
        
        training_end = time.time()
        train_time = training_end - training_start
        
        return best_lam, extract_time, train_time

    def eval_task(self, task_id: int, test_loader):
        test_embeddings, test_labels = feature_extract(self.backbone, test_loader, self.device)
        
        # Project using the CURRENT task's R matrix and WTA
        # FIX Lỗi 7: Không normalize sau WTA, giống hệt FLY-CL
        test_embeddings = self.soho(test_embeddings, self.coding_level, absolute_wta=False)
        
        # Inference
        output = test_embeddings @ self.Wo
        
        predicts = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].squeeze()
        test_accuracy = np.mean(predicts.cpu().numpy() == test_labels.cpu().numpy()) * 100
        return test_accuracy
