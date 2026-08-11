import torch
import numpy as np
from methods.base_cl import BaseCL
from models.soho import SOHO
from utils.train_utils import feature_extract, target2onehot

# Optional: Diagnostic logger (tắt khi không cần)
try:
    from utils.diagnostic_logger import DiagnosticLogger
    _DIAGNOSTIC_AVAILABLE = True
except ImportError:
    _DIAGNOSTIC_AVAILABLE = False

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
        self.G_global = None  # Expose for diagnostic logger
        
        # In SOHO, since projection matrix R changes dynamically per task, 
        # and WTA is a non-linear operation, we must store the backbone features 
        # to re-project and re-solve Ridge accurately at each step.
        self.memory_features = []
        self.memory_labels = []
        
        # Diagnostic logger — gán từ bên ngoài: agent._logger = DiagnosticLogger()
        self._logger = None

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
        n_samples  = all_features.shape[0]

        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                end        = min(i + chunk_size, n_samples)
                feat_chunk = all_features[i:end]
                Y_chunk    = Y[i:end]

                z_chunk = self.soho(feat_chunk, self.coding_level, absolute_wta=False)

                # Cộng dồn Gram matrix và cross-covariance
                Q_global += z_chunk.T @ Y_chunk
                G_global += z_chunk.T @ z_chunk

        # FIX Bug#3: GCV trên sample ngẫu nhiên từ TOÀN BỘ data (không chỉ task mới)
        # -----------------------------------------------------------------------
        # Vấn đề cũ: R thay đổi sau mỗi task → features cũ re-projected qua R mới
        # có phân phối khác task mới → GCV chỉ trên task mới chọn lambda sai.
        # Fix: Sample 3000 mẫu ngẫu nhiên từ all_features đã được chiếu qua R mới,
        # đây là proxy chính xác hơn cho G_global → lambda tốt hơn → Wo tốt hơn.
        gcv_size = min(3000, n_samples)
        gcv_idx  = torch.randperm(n_samples, device=self.device)[:gcv_size]
        with torch.no_grad():
            z_gcv = self.soho(all_features[gcv_idx], self.coding_level, absolute_wta=False)
        Y_gcv    = Y[gcv_idx]
        best_lam = select_ridge_parameter(z_gcv, Y_gcv, self.ridge_lower, self.ridge_upper)
        
        G_reg = G_global + best_lam * torch.eye(G_global.size(0), device=self.device)
        L = torch.linalg.cholesky(G_reg)
        self.Wo = torch.cholesky_solve(Q_global, L)
        self.G_global = G_global  # Expose for diagnostic
        
        training_end = time.time()
        train_time = training_end - training_start
        
        # Gọi diagnostic logger nếu được gắn vào
        if self._logger is not None:
            self._logger.log_task(
                task_id, self.soho, self,
                new_embeddings, new_labels, best_lam, None
            )
        
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
