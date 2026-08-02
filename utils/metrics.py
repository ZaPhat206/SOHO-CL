import numpy as np

def print_accuracy_matrix(acc_dict, num_tasks):
    acc_matrix = [["{:.2f}".format(0.00) for _ in range(num_tasks)] for _ in range(len(acc_dict))]
    for i, (task, values) in enumerate(acc_dict.items()):
        for j, value in enumerate(values):
            acc_matrix[i][i + j] = round(value, 2)
    
    print("\nAccuracy Matrix")
    for row in acc_matrix:
        print(row)
    print()

    print("Average Accuracy")
    A_t = []
    for j in range(num_tasks):
        cnt = 0.0
        for i in range(j + 1):
            cnt += acc_matrix[i][j]
        cnt /= (j + 1)
        A_t.append(cnt)
        print(round(cnt, 2), end=", ")
    print("\n")

    print("Accumulated Accuracy")
    aa = round(np.mean(A_t), 2)
    print(aa)
    print()
    
    # 2.3 Learning Accuracy (LA)
    # Độ chính xác của task i ngay sau khi vừa học xong task i (đường chéo của ma trận)
    la_list = [acc_matrix[i][i] for i in range(num_tasks)]
    la = round(np.mean(la_list), 2)
    print("Learning Accuracy (LA):", la)
    
    # 2.2 Forgetting & 2.4 Backward Transfer (BWT)
    # Tính Forgetting ở bước cuối cùng (sau khi học xong tất cả các task)
    if num_tasks > 1:
        forgetting_list = []
        for i in range(num_tasks - 1): # Bỏ qua task cuối vì chưa bị quên
            # max độ chính xác của task i trong suốt quá trình học từ i đến T-1
            max_acc_past = max([acc_matrix[i][j] for j in range(i, num_tasks - 1)])
            # độ chính xác hiện tại ở bước T
            current_acc = acc_matrix[i][num_tasks - 1]
            forgetting_list.append(max_acc_past - current_acc)
        
        final_forgetting = round(np.mean(forgetting_list), 2)
        bwt = -final_forgetting
        print(f"Forgetting (F): {final_forgetting:.2f}%")
        print(f"Backward Transfer (BWT): {bwt:.2f}%")
    else:
        print("Forgetting (F): 0.00%")
        print("Backward Transfer (BWT): 0.00%")
    print()

    return aa

def compute_memory_footprint(agent):
    """
    Tính tổng dung lượng bộ nhớ (MB) của các ma trận cần lưu trữ (không tính backbone vì frozen)
    """
    total_bytes = 0
    if hasattr(agent, 'Q_global') and agent.Q_global is not None:
        total_bytes += agent.Q_global.element_size() * agent.Q_global.nelement()
    if hasattr(agent, 'G_global') and agent.G_global is not None:
        total_bytes += agent.G_global.element_size() * agent.G_global.nelement()
    if hasattr(agent, 'soho') and hasattr(agent.soho, 'R'):
        total_bytes += agent.soho.R.element_size() * agent.soho.R.nelement()
    if hasattr(agent, 'flyhash') and hasattr(agent.flyhash, 'projection_matrix'):
        # Sparse matrix memory
        mat = agent.flyhash.projection_matrix
        if mat.is_sparse:
            total_bytes += mat.values().element_size() * mat.values().nelement()
            total_bytes += mat.indices().element_size() * mat.indices().nelement()
        else:
            total_bytes += mat.element_size() * mat.nelement()
            
    mb = total_bytes / (1024 * 1024)
    print(f"Memory Footprint (excluding frozen backbone): {mb:.2f} MB\n")
    return mb

def print_timing_metrics(training_time, feature_extract_time):
    print("Training Time")
    for task_time in training_time:
        print(round(task_time, 2), end=", ")
    print("\n")

    print("Average Training Time")
    print(round(np.mean(training_time), 2))
    print()

    print("Feature Extract Time")
    for task_time in feature_extract_time:
        print(round(task_time, 2), end=", ")
    print("\n")

    print("Average Feature Extract Time")
    print(round(np.mean(feature_extract_time), 2))
    print()
