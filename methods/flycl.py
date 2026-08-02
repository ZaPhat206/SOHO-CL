import torch
import numpy as np
from methods.base_cl import BaseCL
from models.flyhash import FlyHash
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
        gcv = (residual / n_samples) / (1 - df / n_samples)**2
        gcv_scores.append(gcv.item())

    optimal_idx = np.argmin(gcv_scores)
    return ridges[optimal_idx]

class FlyCL(BaseCL):
    def __init__(self, backbone, flyhash: FlyHash, num_classes: int, coding_level: float, 
                 ridge_lower: float, ridge_upper: float, device: torch.device):
        super().__init__(backbone, device)
        self.flyhash = flyhash.to(device)
        self.flyhash.to_sparse()
        self.num_classes = num_classes
        self.coding_level = coding_level
        self.ridge_lower = ridge_lower
        self.ridge_upper = ridge_upper
        
        self.Q_global = torch.zeros(flyhash.expand_dim, num_classes, device=device)
        self.G_global = torch.zeros(flyhash.expand_dim, flyhash.expand_dim, device=device)
        self.Wo = None

    def train_task(self, task_id: int, train_loader):
        import time
        training_start = time.time()
        
        feature_extract_start = time.time()
        # Extract features from backbone
        train_embeddings, train_labels = feature_extract(self.backbone, train_loader, self.device)
        feature_extract_end = time.time()
        extract_time = feature_extract_end - feature_extract_start
        
        # Apply sparse random projection
        train_embeddings = self.flyhash(train_embeddings, self.coding_level, absolute_wta=False)
        
        # Accumulate statistics
        Y = target2onehot(train_labels, self.num_classes)
        self.Q_global += train_embeddings.T @ Y
        self.G_global += train_embeddings.T @ train_embeddings
        
        # Select best lambda using GCV
        best_lam = select_ridge_parameter(train_embeddings, Y, self.ridge_lower, self.ridge_upper)
        
        # Solve Ridge Regression
        G_reg = self.G_global + best_lam * torch.eye(self.G_global.size(0), device=self.device)
        L = torch.linalg.cholesky(G_reg)
        self.Wo = torch.cholesky_solve(self.Q_global, L)
        
        training_end = time.time()
        train_time = training_end - training_start
        
        return best_lam, extract_time, train_time

    def eval_task(self, task_id: int, test_loader):
        test_embeddings, test_labels = feature_extract(self.backbone, test_loader, self.device)
        test_embeddings = self.flyhash(test_embeddings, self.coding_level, absolute_wta=False)
        
        # Use sparse matrix multiplication for inference if embeddings are sparsified
        test_embeddings_sparse = test_embeddings.to_sparse_csc()
        output = torch.sparse.mm(test_embeddings_sparse, self.Wo)
        
        predicts = torch.topk(output, k=1, dim=1, largest=True, sorted=True)[1].squeeze()
        test_accuracy = np.mean(predicts.cpu().numpy() == test_labels.cpu().numpy()) * 100
        return test_accuracy
