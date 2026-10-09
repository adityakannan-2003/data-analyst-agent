import pytest


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text("city,sales\nParis,10\nLyon,5\nParis,7\n")
    return path
