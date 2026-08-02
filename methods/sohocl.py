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
        
        # 4. Transform all accumulated features with updated R + WTA
        z_sparse = self.soho(all_features, self.coding_level, absolute_wta=True)
        
        # 5. Recompute Q_global and G_global from scratch for the new subspace
        Y = target2onehot(all_labels, self.num_classes)
        Q_global = z_sparse.T @ Y
        G_global = z_sparse.T @ z_sparse
        
        # 6. Select Ridge Parameter
        best_lam = select_ridge_parameter(z_sparse, Y, self.ridge_lower, self.ridge_upper)
        
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
        
        # Inference
        output = test_embeddings @ self.Wo
        
        predicts = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].squeeze()
        test_accuracy = np.mean(predicts.cpu().numpy() == test_labels.cpu().numpy()) * 100
        return test_accuracy
