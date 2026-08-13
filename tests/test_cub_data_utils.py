from pathlib import Path

import pytest

from utils.data_utils import resolve_cub200_directory


@pytest.mark.parametrize(
    "relative",
    (Path("."), Path("cub"), Path("cub-200-2011"), Path("cub-200-2011/cub")),
)
def test_resolve_cub200_directory_accepts_supported_layouts(tmp_path, relative):
    expected = tmp_path / relative
    (expected / "train").mkdir(parents=True)
    (expected / "test").mkdir()

    assert Path(resolve_cub200_directory(str(tmp_path))).resolve() == expected.resolve()


def test_resolve_cub200_directory_fails_without_complete_split(tmp_path):
    (tmp_path / "cub" / "train").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="exactly one"):
        resolve_cub200_directory(str(tmp_path))
