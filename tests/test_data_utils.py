from utils.data_utils import resolve_cifar100_directory


def test_resolve_cifar_direct_layout(tmp_path):
    for name in ("meta", "train", "test"):
        (tmp_path / name).write_bytes(b"")
    assert resolve_cifar100_directory(str(tmp_path)) == str(tmp_path)


def test_resolve_cifar_parent_layout(tmp_path):
    assert resolve_cifar100_directory(str(tmp_path)).endswith("cifar-100")
