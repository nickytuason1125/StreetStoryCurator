from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np


def test_aadb_is_a_feature():
    import master_judge as mj
    assert "AADB" in mj.FEATURES


def test_feature_vector_reads_aadb_when_present():
    import master_judge as mj
    bd = {"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5,
          "Narrative": 0.5, "Human/Culture": 0.5, "AADB": 0.8}
    vec = mj.feature_vector(bd)
    idx = mj.FEATURES.index("AADB")
    assert vec[idx] == 0.8


def test_feature_vector_is_nan_when_aadb_missing():
    import master_judge as mj
    bd = {"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5,
          "Narrative": 0.5, "Human/Culture": 0.5}  # no "AADB" key
    vec = mj.feature_vector(bd)
    idx = mj.FEATURES.index("AADB")
    assert np.isnan(vec[idx])
