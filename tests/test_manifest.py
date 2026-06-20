from pathlib import Path

from lesion_candidate.data import parse_oct_name


def test_parse_oct_name() -> None:
    meta = parse_oct_name(Path("P001_OS_pre_vol0_slice0007.png"))
    assert meta["patient"] == "P001"
    assert meta["case"] == "001"
    assert meta["eye"] == "OS"
    assert meta["phase"] == "pre"
    assert meta["vol"] == 0
    assert meta["slice"] == 7
