# SRQ-FLY Priority 2D Colab runbook

Run `notebooks/srq_fly_priority2d_cifar_train_only_colab.ipynb` on a Tesla T4.

1. Edit only repository/path values in cell 2.
2. Run every cell in order.
3. The notebook downloads CIFAR-100 and the locked ViT checkpoint.
4. If no local train cache is present, it extracts the 50,000 training images.
   It never creates or loads `test.pt`.
5. The equivalence cell prepares one shared WTA cache, then runs Priority 2B and
   Priority 2C in isolated processes with live task progress.
6. Do not change the split, seed, Ridge value, representation or gates.
7. Download `srq_fly_priority2d_train_only.zip` and return it for audit.

This is the last pre-test gate.  A pass is followed by a separate locked final
test notebook for CIFAR-100, CUB-200-2011 and ImageNet-R.
