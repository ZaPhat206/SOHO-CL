"""
SOHO-CL Diagnostic Logger
==========================
Mục đích: Ghi lại chi tiết nội bộ của quá trình train SOHO-CL sau mỗi Task
để phân tích điểm yếu và đề xuất cải tiến.

Cách dùng trên Kaggle:
    from utils.diagnostic_logger import DiagnosticLogger
    logger = DiagnosticLogger(verbose=True)
    # Rồi truyền logger vào train_task
"""

import torch
import numpy as np

class DiagnosticLogger:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.logs = []  # Lưu toàn bộ log theo tasks

    def log_task(self, task_id, soho_model, agent, new_embeddings, new_labels, best_lam, z_sparse_all):
        """
        Gọi hàm này SAU mỗi train_task để ghi lại toàn bộ trạng thái.
        
        Args:
            task_id:        ID của task vừa train xong
            soho_model:     Object SOHO (có soho.olda, soho.R, soho.W)
            agent:          Object SOHOCL (có memory_features, G_global, Q_global, Wo)
            new_embeddings: Features 768D của task mới (N, 768)
            new_labels:     Labels của task mới
            best_lam:       Lambda được GCV chọn
            z_sparse_all:   Features thưa sau WTA của toàn bộ memory (N_all, 10000)
        """
        olda = soho_model.olda
        n_classes = len(olda.class_sums)
        n_total   = sum(olda.class_counts.values())

        diag = {"task_id": task_id}

        # ==============================================================
        # NHÓM 1: OLDA Discrimination Quality
        # Mục đích: Kiểm tra liệu OLDA có thực sự phân biệt được class không
        # ==============================================================
        with torch.no_grad():
            S_w_norm = olda.S_w / n_total
            S_b = torch.zeros_like(olda.S_w)
            mu_global = olda.global_sum / olda.global_count
            for c, s in olda.class_sums.items():
                n_c = olda.class_counts[c]
                mu_c = s / n_c
                d = (mu_c - mu_global).unsqueeze(1)
                S_b += n_c * (d @ d.T)
            S_b_norm = S_b / n_total

            # Fisher Criterion: tr(S_b) / tr(S_w) — cao = phân biệt tốt
            tr_sb = torch.trace(S_b_norm).item()
            tr_sw = torch.trace(S_w_norm).item()
            fisher_ratio = tr_sb / (tr_sw + 1e-8)

            # Condition number S_w — cao = bất ổn định
            S_w_reg = S_w_norm + 1e-4 * torch.eye(olda.in_dim, device=olda.device)
            try:
                sv_sw = torch.linalg.svdvals(S_w_reg)
                cond_sw = (sv_sw[0] / sv_sw[-1].clamp(min=1e-10)).item()
            except:
                cond_sw = float('nan')

            diag["olda_fisher_ratio"]   = round(fisher_ratio, 4)
            diag["olda_cond_Sw"]        = round(cond_sw, 2)
            diag["olda_tr_Sb"]          = round(tr_sb, 6)
            diag["olda_tr_Sw"]          = round(tr_sw, 6)
            diag["olda_n_classes_seen"] = n_classes

        # ==============================================================
        # NHÓM 2: Projection Matrix R — Stability
        # Mục đích: R thay đổi bao nhiêu so với task trước?
        # Nếu thay đổi quá nhiều → Re-projection làm mất thông tin cũ
        # ==============================================================
        R_current = soho_model.R  # (olda_dim, 768)
        diag["R_frobenius_norm"] = round(torch.linalg.norm(R_current).item(), 4)

        if hasattr(self, "_R_prev") and self._R_prev is not None:
            delta_R = torch.linalg.norm(R_current - self._R_prev).item()
            R_change_ratio = delta_R / (torch.linalg.norm(self._R_prev).item() + 1e-8)
            diag["R_change_ratio"] = round(R_change_ratio, 4)
        else:
            diag["R_change_ratio"] = None

        self._R_prev = R_current.clone()

        # ==============================================================
        # NHÓM 3: WTA Winner Neurons
        # Mục đích: Các task có "tranh giành" các neurons thắng không?
        # Overlap cao = Interference = Forgetting
        # ==============================================================
        with torch.no_grad():
            # Lấy winner neurons của task MỚI
            new_emb_norm = torch.nn.functional.normalize(new_embeddings, p=2, dim=1)
            z_new = new_emb_norm @ soho_model.R.T
            expanded_new = soho_model.W @ z_new.T  # (output_dim, N)
            k = max(1, int(expanded_new.shape[0] * 0.45))  # Dùng coding_level mặc định
            _, idx_new = expanded_new.topk(k, dim=0)
            winners_new = set(idx_new.flatten().cpu().numpy().tolist())

            diag["wta_n_unique_winners_new_task"] = len(winners_new)

            # Overlap với task trước (nếu có)
            if hasattr(self, "_winners_prev") and self._winners_prev is not None:
                overlap = len(winners_new & self._winners_prev)
                overlap_ratio = overlap / (len(winners_new) + 1e-8)
                diag["wta_overlap_with_prev_task"] = round(overlap_ratio, 4)
            else:
                diag["wta_overlap_with_prev_task"] = None

            self._winners_prev = winners_new

        # ==============================================================
        # NHÓM 4: Ridge Regression — G Matrix Health
        # Mục đích: Gram matrix G có bị ill-conditioned không?
        # Condition number tăng liên tục = dấu hiệu của scale drift
        # ==============================================================
        with torch.no_grad():
            if agent.G_global is not None:
                try:
                    sv_G = torch.linalg.svdvals(agent.G_global)
                    # Dùng effective rank thay vì full cond (nhanh hơn)
                    sv_normalized = sv_G / (sv_G[0] + 1e-10)
                    effective_rank = (sv_normalized > 1e-4).sum().item()
                    cond_G = (sv_G[0] / sv_G[effective_rank-1].clamp(min=1e-10)).item()
                    diag["gram_cond_number"]   = round(cond_G, 2)
                    diag["gram_effective_rank"] = effective_rank
                    diag["gram_trace"]          = round(torch.trace(agent.G_global).item(), 4)
                except:
                    diag["gram_cond_number"] = float("nan")
            diag["ridge_best_lambda"] = best_lam.item() if hasattr(best_lam, 'item') else best_lam

        # ==============================================================
        # NHÓM 5: ETF Alignment Quality
        # Mục đích: ETF Procrustes có thực sự căn chỉnh được class centroids không?
        # Cosine similarity giữa projected centroids và ETF target
        # ==============================================================
        with torch.no_grad():
            try:
                centroids_proj = []
                for c in sorted(olda.class_sums.keys()):
                    mu_c = olda.class_sums[c] / olda.class_counts[c]
                    mu_c_norm = torch.nn.functional.normalize(mu_c.unsqueeze(0), p=2, dim=1)
                    c_proj = mu_c_norm @ soho_model.R.T  # (1, olda_dim)
                    centroids_proj.append(c_proj)

                C_mat = torch.cat(centroids_proj, dim=0)  # (N_classes, olda_dim)
                C_norm = torch.nn.functional.normalize(C_mat, p=2, dim=1)
                # Cosine similarity matrix giữa các class centroids
                sim_mat = C_norm @ C_norm.T
                # Lý tưởng ETF: tất cả off-diagonal = -1/(N-1)
                etf_target_sim = -1.0 / (n_classes - 1) if n_classes > 1 else 0.0
                off_diag_mask = ~torch.eye(n_classes, dtype=torch.bool, device=olda.device)
                actual_mean_sim = sim_mat[off_diag_mask].mean().item()
                etf_deviation = abs(actual_mean_sim - etf_target_sim)

                diag["etf_target_cosine"]   = round(etf_target_sim, 4)
                diag["etf_actual_cosine"]   = round(actual_mean_sim, 4)
                diag["etf_deviation"]       = round(etf_deviation, 4)
            except Exception as e:
                diag["etf_deviation"] = float("nan")

        n_memory = sum(f.shape[0] for f in agent.memory_features)
        mem_mb = n_memory * 768 * 4 / (1024**2)
        diag["memory_n_vectors"] = n_memory
        diag["memory_mb_768d"]   = round(mem_mb, 2)

        # ==============================================================
        # NHÓM 7: Feature Distribution Shift (Quan trọng nhất!)
        # Mục đích: Khi R cập nhật, features của task CŨ bị dịch chuyển bao nhiêu?
        # Nếu lớn → đây là nguồn gốc trực tiếp của Forgetting trong SOHO-CL
        # ==============================================================
        with torch.no_grad():
            if len(agent.memory_features) > 1 and hasattr(self, "_R_prev") and self._R_prev is not None:
                try:
                    X_oldest = agent.memory_features[0]  # Task 0, cụm cũ nhất
                    X_norm = torch.nn.functional.normalize(X_oldest, p=2, dim=1)
                    Z_prev = X_norm @ self._R_prev_before_update.T  # Projection cũ
                    Z_curr = X_norm @ soho_model.R.T                # Projection mới
                    Z_prev_n = torch.nn.functional.normalize(Z_prev, p=2, dim=1)
                    Z_curr_n = torch.nn.functional.normalize(Z_curr, p=2, dim=1)
                    cosine_sim = (Z_prev_n * Z_curr_n).sum(dim=1).mean().item()
                    shift_score = 1.0 - cosine_sim  # 0=không dịch, 1=lật hẳn
                    diag["feature_shift_oldest_task"] = round(shift_score, 4)
                except:
                    diag["feature_shift_oldest_task"] = float("nan")
            else:
                diag["feature_shift_oldest_task"] = None

        # Lưu R trước khi bị ghi đè (dùng ở task tiếp theo)
        self._R_prev_before_update = soho_model.R.clone()

        # ==============================================================
        # NHÓM 8: Null-Space Leakage
        # Mục đích: Kiểm tra NSP có thực sự bảo vệ kiến thức cũ không?
        # Nếu old features chiếu lên discriminative dir mới ≈ 0 → NSP đang hoạt động
        # ==============================================================
        with torch.no_grad():
            if len(agent.memory_features) > 1:
                try:
                    n_disc = n_classes - 1
                    if n_disc > 0:
                        R_disc = soho_model.R[:n_disc, :]  # Discriminative subspace (n_disc, 768)
                        X_old = agent.memory_features[0]
                        X_old_norm = torch.nn.functional.normalize(X_old, p=2, dim=1)  # (N, 768)
                        # Chiếu features cũ lên discriminative directions mới
                        proj_on_disc = X_old_norm @ R_disc.T  # (N, n_disc)
                        # Leakage = tỷ lệ năng lượng rời vào discriminative dir
                        energy_disc = proj_on_disc.norm(dim=1).mean().item()
                        energy_total = (X_old_norm @ soho_model.R.T).norm(dim=1).mean().item()
                        leakage = energy_disc / (energy_total + 1e-8)
                        diag["nsp_leakage_ratio"] = round(leakage, 4)
                    else:
                        diag["nsp_leakage_ratio"] = None
                except:
                    diag["nsp_leakage_ratio"] = float("nan")
            else:
                diag["nsp_leakage_ratio"] = None

        # ==============================================================
        # NHÓM 9: Ridge Regression Residual
        # Mục đích: Wo có giải tốt không? Nếu residual tăng → Ridge đang overfit task mới
        # ==============================================================
        with torch.no_grad():
            if agent.Wo is not None and agent.G_global is not None:
                try:
                    # Sample nhanh 500 mẫu để ước lượng residual
                    all_feats = torch.cat(agent.memory_features, dim=0)
                    all_labs  = torch.cat(agent.memory_labels, dim=0)
                    sample_n  = min(500, all_feats.shape[0])
                    idx = torch.randperm(all_feats.shape[0])[:sample_n]
                    from utils.train_utils import target2onehot
                    Y_sample = target2onehot(all_labs[idx], agent.num_classes)
                    H_sample = agent.soho(all_feats[idx], agent.coding_level)
                    Y_hat    = H_sample @ agent.Wo
                    residual = torch.nn.functional.mse_loss(Y_hat, Y_sample).item()
                    diag["ridge_mse_residual"] = round(residual, 6)
                except:
                    diag["ridge_mse_residual"] = float("nan")
            else:
                diag["ridge_mse_residual"] = None

        # ==============================================================
        # Lưu và In
        # ==============================================================
        self.logs.append(diag)

        if self.verbose:
            print(f"\n{'='*65}")
            print(f"  📊 DIAGNOSTIC — Task {task_id:02d}")
            print(f"{'='*65}")
            print(f"  [OLDA]  Fisher Ratio     = {diag['olda_fisher_ratio']:>10.4f}  (cao > tốt)")
            print(f"  [OLDA]  Cond(S_w)        = {diag['olda_cond_Sw']:>10.2f}  (thấp > ổn định)")
            print(f"  [R]     Change Ratio      = {str(diag['R_change_ratio']):>10}  (thấp > ổn định)")
            print(f"  [WTA]   Winner Overlap    = {str(diag['wta_overlap_with_prev_task']):>10}  (thấp > tốt)")
            print(f"  [GRAM]  Cond(G)           = {diag.get('gram_cond_number', 'N/A'):>10}  (thấp > ổn định)")
            print(f"  [GRAM]  Best Lambda       = {diag['ridge_best_lambda']:>10}  (ổn định > không nhảy)")
            print(f"  [ETF]   ETF Deviation     = {diag['etf_deviation']:>10.4f}  (thấp > tốt)")
            print(f"  [SHIFT] Feature Shift     = {str(diag.get('feature_shift_oldest_task','N/A')):>10}  (<0.1 > tốt)")
            print(f"  [NSP]   Leakage Ratio     = {str(diag.get('nsp_leakage_ratio','N/A')):>10}  (<0.3 > tốt)")
            print(f"  [RIDGE] MSE Residual      = {str(diag.get('ridge_mse_residual','N/A')):>10}  (ổn định > tốt)")
            print(f"  [MEM]   Memory (768D)     = {diag['memory_mb_768d']:>8.2f} MB")
            print(f"{'='*65}\n")

        return diag

    def summary(self):
        """In bảng tổng kết toàn bộ quá trình train."""
        print("\n" + "="*95)
        print("📋 DIAGNOSTIC SUMMARY — Toàn Bộ Quá Trình Train")
        print("="*95)
        print(f"{'Task':>4} | {'Fisher':>8} | {'Cond(Sw)':>9} | {'R_chg':>6} | {'WTA_ovlp':>8} | {'Cond(G)':>9} | {'Lambda':>8} | {'Shift':>6} | {'Leakage':>7} | {'MSE':>8}")
        print("-"*95)
        for d in self.logs:
            print(
                f"  {d['task_id']:>2d} | "
                f"{d['olda_fisher_ratio']:>8.4f} | "
                f"{d['olda_cond_Sw']:>9.1f} | "
                f"{str(d['R_change_ratio'] or 'N/A'):>6} | "
                f"{str(d['wta_overlap_with_prev_task'] or 'N/A'):>8} | "
                f"{str(d.get('gram_cond_number','N/A')):>9} | "
                f"{d['ridge_best_lambda']:>8.0f} | "
                f"{str(d.get('feature_shift_oldest_task','N/A')):>6} | "
                f"{str(d.get('nsp_leakage_ratio','N/A')):>7} | "
                f"{str(d.get('ridge_mse_residual','N/A')):>8}"
            )
        print("="*95)
        print("\nℹ️  Phân tích nhanh:")
        shifts = [d['feature_shift_oldest_task'] for d in self.logs if d.get('feature_shift_oldest_task') is not None and d['feature_shift_oldest_task'] == d['feature_shift_oldest_task']]
        leakages = [d['nsp_leakage_ratio'] for d in self.logs if d.get('nsp_leakage_ratio') is not None and d['nsp_leakage_ratio'] == d['nsp_leakage_ratio']]
        if shifts:
            print(f"   Feature Shift trung bình: {np.mean(shifts):.4f} (ngưỡng nguy hiểm: >0.1)")
        if leakages:
            print(f"   NSP Leakage trung bình:   {np.mean(leakages):.4f} (ngưỡng nguy hiểm: >0.3)")

