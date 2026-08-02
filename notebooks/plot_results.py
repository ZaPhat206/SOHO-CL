import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(acc_matrix):
    """
    Computes Average Accuracy and Forgetting given a strictly lower triangular 
    (or jagged) accuracy matrix.
    acc_matrix[i][j] is the accuracy of task j after learning task i (i >= j).
    """
    num_tasks = len(acc_matrix)
    average_accuracy = []
    forgetting = []

    for t in range(num_tasks):
        # Average accuracy at task t
        aa_t = np.mean([acc_matrix[t][j] for j in range(t + 1)])
        average_accuracy.append(aa_t)
        
        # Forgetting at task t (only makes sense for t > 0)
        if t == 0:
            forgetting.append(0.0)
        else:
            f_t = np.mean([max([acc_matrix[i][j] for i in range(j, t)]) - acc_matrix[t][j] for j in range(t)])
            forgetting.append(f_t)

    return average_accuracy, forgetting

def plot_comparisons(flycl_matrix, sohocl_matrix, dataset_name="CIFAR-100"):
    fly_aa, fly_f = compute_metrics(flycl_matrix)
    soho_aa, soho_f = compute_metrics(sohocl_matrix)
    
    tasks = np.arange(1, len(fly_aa) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Average Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(tasks, fly_aa, marker='o', label=f'FLY-CL (Final: {fly_aa[-1]:.2f}%)', color='#3498db', linewidth=2)
    plt.plot(tasks, soho_aa, marker='s', label=f'SOHO-CL (Final: {soho_aa[-1]:.2f}%)', color='#e74c3c', linewidth=2)
    plt.title(f'{dataset_name} - Average Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.xticks(tasks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # Plot Forgetting
    plt.subplot(1, 2, 2)
    plt.plot(tasks, fly_f, marker='o', label=f'FLY-CL (Final: {fly_f[-1]:.2f}%)', color='#3498db', linewidth=2)
    plt.plot(tasks, soho_f, marker='s', label=f'SOHO-CL (Final: {soho_f[-1]:.2f}%)', color='#e74c3c', linewidth=2)
    plt.title(f'{dataset_name} - Forgetting', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Tasks Learned', fontsize=12)
    plt.ylabel('Average Forgetting (%)', fontsize=12)
    plt.xticks(tasks)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage in Kaggle:
if __name__ == "__main__":
    print("Replace these with the actual Accuracy Matrices printed in the console.")
    # flycl_matrix = [[...], [...], ...]
    # sohocl_matrix = [[...], [...], ...]
    # plot_comparisons(flycl_matrix, sohocl_matrix)
