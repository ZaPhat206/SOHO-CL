import subprocess
import pandas as pd
from IPython.display import display, Markdown

# ===========================================================================
# DATASET CONFIGS — mỗi dataset có profile riêng, không share mặc định CIFAR
# ===========================================================================
DATASET_PROFILES = {
    "CIFAR-100": {
        "num_classes":  100,
        "num_tasks":    10,          # 10 tasks = 10 class/task (chuẩn paper)
        "model_name":   "vit_base_patch16_224",   # IN-1K đủ cho CIFAR
        "data_aug":     "none",                   # CIFAR không cần normalize
        "flycl":  {"cl": 0.3,  "r_lower": 4,  "expand_dim": 10000, "density": 0.3},
        "sohocl": {"cl": 0.25, "r_lower": -2, "expand_dim": 10000, "density": 0.1},
    },
    "CUB-200-2011": {
        "num_classes":  200,
        "num_tasks":    10,          # 10 tasks = 20 class/task
        "model_name":   "vit_base_patch16_224",
        "data_aug":     "vit",       # CUB cần chuẩn hóa màu ViT style
        "flycl":  {"cl": 0.3,  "r_lower": 4,  "expand_dim": 10000, "density": 0.3},
        "sohocl": {"cl": 0.2,  "r_lower": -2, "expand_dim": 10000, "density": 0.3},
    },
    "ImageNet-R": {
        "num_classes":  200,
        "num_tasks":    10,          # 10 tasks = 20 class/task (chuẩn paper)
        "model_name":   "vit_base_patch16_224_in21k",  # BẮT BUỘC IN-21K!
        "data_aug":     "vit",
        "flycl":  {"cl": 0.3,  "r_lower": 4,  "expand_dim": 10000, "density": 0.3},
        "sohocl": {"cl": 0.25, "r_lower": -2, "expand_dim": 10000, "density": 0.1},
    },
}

def run_experiment(method, dataset, num_tasks=None, olda_dim=768, use_etf=True):
    """
    Chạy một thử nghiệm Continual Learning với config tối ưu cho từng dataset.
    
    Args:
        method:    'flycl' hoặc 'sohocl'
        dataset:   'CIFAR-100', 'CUB-200-2011', 'ImageNet-R'
        num_tasks: Override số task (để None = dùng default chuẩn paper)
        olda_dim:  Chiều OLDA cho SOHO-CL (mặc định 768)
        use_etf:   Có dùng ETF Procrustes alignment hay không
    """
    if dataset not in DATASET_PROFILES:
        raise ValueError(f"Dataset không hỗ trợ: {dataset}. Chọn: {list(DATASET_PROFILES.keys())}")
    
    profile    = DATASET_PROFILES[dataset]
    method_cfg = profile[method]
    
    num_classes = profile["num_classes"]
    n_tasks     = num_tasks if num_tasks is not None else profile["num_tasks"]
    model_name  = profile["model_name"]
    data_aug    = profile["data_aug"]
    
    cl          = method_cfg["cl"]
    r_lower     = method_cfg["r_lower"]
    expand_dim  = method_cfg["expand_dim"]
    density     = method_cfg["density"]

    print(f"\n{'='*60}")
    print(f"⏳ {method.upper()} trên {dataset}")
    print(f"   Tasks={n_tasks} | Classes={num_classes} | Model={model_name}")
    print(f"   coding_level={cl} | density={density} | ridge_lower={r_lower}")
    print(f"   data_aug={data_aug} | OLDA={olda_dim}D | ETF={'ON' if use_etf else 'OFF'}")
    print(f"{'='*60}")
    
    root_path = "/kaggle/input/datasets/zaphat206"

    cmd = [
        "python", "main.py",
        "--method",        method,
        "--dataset",       dataset,
        "--num_classes",   str(num_classes),
        "--num_tasks",     str(n_tasks),
        "--model_name",    model_name,      # QUAN TRỌNG: backbone đúng
        "--data_augmentation", data_aug,    # QUAN TRỌNG: normalize đúng
        "--coding_level",  str(cl),
        "--ridge_lower",   str(r_lower),
        "--expand_dim",    str(expand_dim),
        "--density",       str(density),
        "--root",          root_path,
    ]

    if method == 'sohocl':
        cmd.extend(["--olda_dim", str(olda_dim)])
        if not use_etf:
            cmd.append("--no_etf")

    print(f"   [CMD] {' '.join(cmd)}\n")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        output_lines.append(line)
        if "[Task" in line or "Starting Continual" in line or "Evaluation Summary" in line:
            print("   " + line.strip())
    process.stdout.close()
    process.wait()
    output = "".join(output_lines)

    metrics = {"Method": method.upper(), "Dataset": dataset,
               "Tasks": n_tasks, "Model": model_name}
    try:
        metrics["AA (%)"]           = float(output.split("Accumulated Accuracy\n")[1].split("\n")[0])
        metrics["LA (%)"]           = float(output.split("Learning Accuracy (LA): ")[1].split("\n")[0])
        metrics["Forgetting (%)"]   = float(output.split("Forgetting (F): ")[1].split("%")[0])
        metrics["BWT (%)"]          = float(output.split("Backward Transfer (BWT): ")[1].split("%")[0])
        metrics["Memory (MB)"]      = float(output.split("Memory Footprint (excluding frozen backbone): ")[1].split(" MB")[0])
        metrics["Avg Train Time (s)"] = float(output.split("Average Training Time\n")[1].split("\n")[0])
        metrics["_matrix_str"]      = output.split("Accuracy Matrix\n")[1].split("\nAverage Accuracy")[0]
        
        print("\n📊 Accuracy Matrix:")
        print(metrics["_matrix_str"])
        print(f"\n✅ HOÀN TẤT! AA={metrics['AA (%)']}% | F={metrics['Forgetting (%)']}%\n")
    except Exception as e:
        print(f"❌ Parse thất bại: {e}")
        print("--- 10 dòng cuối ---")
        print("".join(output_lines[-10:]))

    return metrics


# ===========================================================================
# ABLATION PLAN — chạy tuần tự theo priority
# ===========================================================================
if __name__ == "__main__":
    all_results = []

    # BƯỚC 1: Chạy baseline FLY-CL (làm mốc so sánh công bằng)
    print("\n" + "🔵 "*20)
    print("BƯỚC 1: FLY-CL BASELINE")
    for ds in ["CIFAR-100", "CUB-200-2011", "ImageNet-R"]:
        r = run_experiment("flycl", ds)
        all_results.append(r)

    # BƯỚC 2: Chạy SOHO-CL với config tối ưu (đã fix bug)
    print("\n" + "🟢 "*20)
    print("BƯỚC 2: SOHO-CL FIXED")
    for ds in ["CIFAR-100", "CUB-200-2011", "ImageNet-R"]:
        r = run_experiment("sohocl", ds)
        all_results.append(r)

    # Hiện bảng so sánh
    df = pd.DataFrame(all_results)
    cols = ["Method", "Dataset", "Tasks", "AA (%)", "LA (%)", "Forgetting (%)", "Memory (MB)", "Avg Train Time (s)"]
    display(df[[c for c in cols if c in df.columns]].to_markdown(index=False))
