from src.eval import *

def test_extract_all_labels():
    assert set(extract_all_labels("toprock, error, none")) == set(["None"])
    assert set(extract_all_labels("powermove, error, footowork, notes: has footwork freezes and toprocks")) == set(["footwork", "toprock"])
    assert set(extract_all_labels("powermove, notes: had a freeze at the end")) == set(["powermove"])
    assert set(extract_all_labels("toprock")) == set(["toprock"])
    assert set(extract_all_labels("powermove, error, freezes")) == set(["None"])