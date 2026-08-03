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
        
        # Ngăn chặn lỗi chia cho 0 khi số chiều D > số mẫu N (khiến df tiến tới n_samples)
        denom = max(1.0 - (df / n_samples).item(), 1e-4)
        gcv = (residual / n_samples) / (denom ** 2)
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
        
        # Bơm Trick 2: Nhãn Mềm (Label Smoothing)
        Y_onehot = target2onehot(all_labels, self.num_classes)
        Y = Y_onehot * 0.95 + 0.05 / self.num_classes
        
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
                
                # Biến đổi và chuẩn hóa L2 cho từng Chunk
                z_chunk = self.soho(feat_chunk, self.coding_level, absolute_wta=True)
                z_chunk = torch.nn.functional.normalize(z_chunk, p=2, dim=1)
                
                # Cộng dồn ma trận (Y hệt FLY-CL)
                Q_global += z_chunk.T @ Y_chunk
                G_global += z_chunk.T @ z_chunk
                
                # Trích xuất riêng dữ liệu của Task mới nhất để tính Lambda
                if end > start_new:
                    chunk_start_in_new = max(0, start_new - i)
                    z_sparse_new_list.append(z_chunk[chunk_start_in_new:])
                
        z_sparse_new = torch.cat(z_sparse_new_list, dim=0)
        Y_new = Y[start_new:]
        
        # 6. Select Ridge Parameter (Chỉ dùng dữ liệu mới nhất để tăng tốc)
        best_lam = select_ridge_parameter(z_sparse_new, Y_new, self.ridge_lower, self.ridge_upper)
        
        # 7. Solve Ridge Regression
        # Thêm 1e-4 * I để đảm bảo ma trận luôn khả nghịch (tránh Singularity khi N <= D)
        G_reg = G_global + (best_lam + 1e-4) * torch.eye(G_global.size(0), device=self.device)
        L = torch.linalg.cholesky(G_reg)
        self.Wo = torch.cholesky_solve(Q_global, L)
        
        training_end = time.time()
        train_time = training_end - training_start
        
        return best_lam, extract_time, train_time

    def eval_task(self, task_id: int, test_loader):
        test_embeddings, test_labels = feature_extract(self.backbone, test_loader, self.device)
        
        # Project using the CURRENT task's R matrix and WTA
        test_embeddings = self.soho(test_embeddings, self.coding_level, absolute_wta=True)
        # Bơm Trick 1: Chuẩn hóa L2 tương tự lúc Train
        test_embeddings = torch.nn.functional.normalize(test_embeddings, p=2, dim=1)
        
        # Inference
        output = test_embeddings @ self.Wo
        
        predicts = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].squeeze()
        test_accuracy = np.mean(predicts.cpu().numpy() == test_labels.cpu().numpy()) * 100
        return test_accuracy
