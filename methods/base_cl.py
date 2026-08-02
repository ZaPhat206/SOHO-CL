import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaseCL(nn.Module):
    def __init__(self, backbone: nn.Module, device: torch.device):
        super().__init__()
        self.backbone = backbone
        self.device = device
        
    def train_task(self, task_id: int, train_loader: DataLoader):
        """Train on a specific task."""
        raise NotImplementedError

    def eval_task(self, task_id: int, test_loader: DataLoader):
        """Evaluate on a specific task."""
        raise NotImplementedError
